import base64
import importlib
import importlib.util
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import gi

gi.require_version("Gtk", "3.0")
gi.require_version('GdkPixbuf', '2.0')
gi.require_version('Gdk', '3.0')
from gi.repository import GLib, Gdk, GdkPixbuf, Gtk, Pango

from google import genai
from google.genai import types


DATA_DIR = Path.home() / ".gemini_gtk"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

CONVERSATIONS_FILE = DATA_DIR / "conversations.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
# Do not alter these model names!
DEFAULT_MODELS = [
    ("gemini-flash-latest", "Gemini Flash"),
    ("gemini-pro-latest", "Gemini Thinking"),
    ("gemini-2.5-flash-image", "Nano Banana"),
    ("gemini-3-pro-image-preview", "Nano Banana Pro"),
]


@dataclass
class Message:
    role: str
    content: str
    images: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Conversation:
    id: str
    title: str
    model: str
    messages: List[Message] = field(default_factory=list)


@dataclass
class Settings:
    font_size: int = 12
    user_color: str = "#1a73e8"
    assistant_color: str = "#1b5e20"
    user_name: str = "User"
    assistant_name: str = "Assistant"


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.settings = Settings()
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        self.settings = Settings(
            font_size=int(data.get("font_size", self.settings.font_size)),
            user_color=data.get("user_color", self.settings.user_color),
            assistant_color=data.get("assistant_color", self.settings.assistant_color),
            user_name=data.get("user_name", self.settings.user_name),
            assistant_name=data.get("assistant_name", self.settings.assistant_name),
        )

    def save(self) -> None:
        payload = {
            "font_size": self.settings.font_size,
            "user_color": self.settings.user_color,
            "assistant_color": self.settings.assistant_color,
            "user_name": self.settings.user_name,
            "assistant_name": self.settings.assistant_name,
        }
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.conversations: List[Conversation] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.conversations = []
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self.conversations = []
            return
        self.conversations = [
            Conversation(
                id=item.get("id", str(uuid.uuid4())),
                title=item.get("title", "Untitled"),
                model=item.get("model", DEFAULT_MODELS[0][0]),
                messages=[
                    Message(
                        role=msg.get("role", ""),
                        content=msg.get("content", ""),
                        images=msg.get("images", []),
                        timestamp=msg.get(
                            "timestamp", datetime.now().isoformat(timespec="seconds")
                        ),
                    )
                    for msg in item.get("messages", [])
                    if msg.get("content") or msg.get("images")
                ],
            )
            for item in data
        ]

    def save(self) -> None:
        payload = [
            {
                "id": c.id,
                "title": c.title,
                "model": c.model,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "images": m.images,
                        "timestamp": m.timestamp,
                    }
                    for m in c.messages
                ],
            }
            for c in self.conversations
        ]
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def create(self, title: str, model: str) -> Conversation:
        convo = Conversation(id=str(uuid.uuid4()), title=title, model=model)
        self.conversations.append(convo)
        self.save()
        return convo

    def delete(self, conversation_id: str) -> None:
        self.conversations = [c for c in self.conversations if c.id != conversation_id]
        self.save()

    def get(self, conversation_id: str) -> Optional[Conversation]:
        for convo in self.conversations:
            if convo.id == conversation_id:
                return convo
        return None


class ModelClient:
    def __init__(self) -> None:
        self.client = None
        self.types = None
        self.configuration_error: Optional[str] = None

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
        if not api_key:
            self.configuration_error = (
                "Set GEMINI_API_KEY or GOOGLE_GENAI_API_KEY to enable real model calls."
            )
            return

        if importlib.util.find_spec("google.genai") is None:
            self.configuration_error = (
                "Install google-genai to enable real model calls: pip install google-genai"
            )
            return

        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")

        self.client = genai.Client(api_key=api_key)
        self.types = types

    def generate_reply(self, conversation: Conversation) -> Tuple[bool, str, List[str]]:
        if not self.client or not self.types:
            return False, self.configuration_error or "API client is not configured.", []

        contents = [
            self.types.Content(
                role="user" if message.role == "user" else "model",
                parts=[self.types.Part(text=message.content)],
            )
            for message in conversation.messages
            if message.content
        ]
        if not contents:
            return False, "Conversation has no messages to send.", []

        try:
            response = self.client.models.generate_content(
                model=conversation.model,
                contents=contents,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"API error: {exc}", []

        text, images = self._extract_candidate_parts(response)
        if not text and not images:
            return False, "Received empty response from the model.", []
        return True, text.strip(), images

    def _extract_candidate_parts(self, response: object) -> Tuple[str, List[str]]:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return "", []

        text_parts: List[str] = []
        images: List[str] = []

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for part in parts:
                text = getattr(part, "text", "")
                if text:
                    text_parts.append(text)
                inline_data = getattr(part, "inline_data", None)
                if inline_data:
                    image_path = self._save_inline_image(inline_data)
                    if image_path:
                        images.append(image_path)
        return "".join(text_parts), images

    def _save_inline_image(self, inline_data: object) -> Optional[str]:
        raw_data = getattr(inline_data, "data", None)
        if raw_data is None:
            return None

        image_bytes: Optional[bytes] = None
        if isinstance(raw_data, (bytes, bytearray, memoryview)):
            image_bytes = bytes(raw_data)
        elif isinstance(raw_data, str):
            try:
                image_bytes = base64.b64decode(raw_data, validate=True)
            except Exception:  # noqa: BLE001
                return None
        else:
            return None

        if not image_bytes:
            return None

        mime_type = getattr(inline_data, "mime_type", "image/png")
        extension = ".png"
        if mime_type == "image/jpeg":
            extension = ".jpg"
        elif mime_type == "image/webp":
            extension = ".webp"

        filename = IMAGES_DIR / f"{uuid.uuid4()}{extension}"
        try:
            with filename.open("wb") as f:
                f.write(image_bytes)
        except OSError:
            return None

        return str(filename)


class ChatWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, store: ConversationStore, settings_store: SettingsStore):
        super().__init__(application=app)
        self.set_default_size(900, 600)
        self.set_title("Gemini GTK")

        self.store = store
        self.settings_store = settings_store
        self.settings = settings_store.settings
        self.selected_conversation: Optional[Conversation] = None
        self.model_client = ModelClient()
        self._mathtext = None

        root_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.add(root_box)

        self.sidebar = self._create_sidebar()
        root_box.pack_start(self.sidebar, False, False, 0)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        root_box.pack_start(content_box, True, True, 0)

        self.message_view = self._create_message_view()
        content_box.pack_start(self.message_view, True, True, 0)

        self.input_bar = self._create_input_bar()
        content_box.pack_start(self.input_bar, False, False, 0)

        if self.store.conversations:
            self._select_conversation(self.store.conversations[0])

    def _create_sidebar(self) -> Gtk.Widget:
        frame = Gtk.Frame()
        frame.set_size_request(250, -1)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=6)
        frame.add(box)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        new_button = Gtk.Button(label="New Chat")
        new_button.connect("clicked", self.on_new_chat)
        delete_button = Gtk.Button(label="Delete")
        delete_button.connect("clicked", self.on_delete_chat)
        settings_button = Gtk.Button(label="Settings")
        settings_button.connect("clicked", self.on_open_settings)
        header.pack_start(new_button, True, True, 0)
        header.pack_start(delete_button, True, True, 0)
        header.pack_start(settings_button, True, True, 0)
        box.pack_start(header, False, False, 0)

        self.listbox = Gtk.ListBox()
        self.listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.listbox.connect("row-selected", self.on_conversation_selected)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.add(self.listbox)
        box.pack_start(scroller, True, True, 0)

        self._refresh_sidebar()
        return frame

    def _create_message_view(self) -> Gtk.Widget:
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self.textview = Gtk.TextView()
        self.textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.textview.set_editable(False)
        self.textbuffer = self.textview.get_buffer()

        self._register_tags()
        scrolled.add(self.textview)
        return scrolled

    def _create_input_bar(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6, margin=6)

        self.model_combo = Gtk.ComboBoxText()
        for value, label in DEFAULT_MODELS:
            self.model_combo.append_text(label)
        self.model_combo.set_active(0)
        box.pack_start(self.model_combo, False, False, 0)

        self.entry = Gtk.Entry()
        self.entry.set_placeholder_text("Ask Gemini or Nano Banana...")
        self.entry.connect("activate", self.on_send)
        box.pack_start(self.entry, True, True, 0)

        send_button = Gtk.Button(label="Send")
        send_button.connect("clicked", self.on_send)
        box.pack_start(send_button, False, False, 0)

        return box

    def _register_tags(self) -> None:
        self.textbuffer.create_tag("role_base", weight=Pango.Weight.BOLD, scale=1.05)
        self.textbuffer.create_tag("user_role")
        self.textbuffer.create_tag("assistant_role")
        self.textbuffer.create_tag("timestamp", foreground="#777777", scale=0.9)
        self.textbuffer.create_tag("user_message", pixels_above_lines=2, pixels_below_lines=6)
        self.textbuffer.create_tag("assistant_message", pixels_above_lines=2, pixels_below_lines=6)
        self.textbuffer.create_tag("heading1", weight=Pango.Weight.BOLD, scale=1.35)
        self.textbuffer.create_tag("heading2", weight=Pango.Weight.BOLD, scale=1.2)
        self.textbuffer.create_tag("heading3", weight=Pango.Weight.BOLD, scale=1.1)
        self.textbuffer.create_tag("bold", weight=Pango.Weight.BOLD)
        self.textbuffer.create_tag("italic", style=Pango.Style.ITALIC)
        self.textbuffer.create_tag("code", family="Monospace", background="#f5f5f5")
        self.textbuffer.create_tag("bullet", left_margin=12)
        self.textbuffer.create_tag("hr", foreground="#999999", pixels_above_lines=6, pixels_below_lines=6)
        self._apply_settings()

    def _apply_settings(self) -> None:
        font_desc = Pango.FontDescription()
        font_desc.set_size(self.settings.font_size * Pango.SCALE)
        self.textview.modify_font(font_desc)

        tag_table = self.textbuffer.get_tag_table()
        for tag_name, color in (
            ("user_role", self.settings.user_color),
            ("assistant_role", self.settings.assistant_color),
            ("user_message", self.settings.user_color),
            ("assistant_message", self.settings.assistant_color),
        ):
            tag = tag_table.lookup(tag_name)
            if tag:
                tag.set_property("foreground", color)

    def _refresh_sidebar(self) -> None:
        for child in self.listbox.get_children():
            self.listbox.remove(child)

        for convo in self.store.conversations:
            row = Gtk.ListBoxRow()
            row.conversation_id = convo.id
            label = Gtk.Label(label=convo.title, xalign=0)
            label.set_line_wrap(True)
            label.set_max_width_chars(30)
            label.set_ellipsize(Pango.EllipsizeMode.END)
            row.add(label)
            self.listbox.add(row)
        self.listbox.show_all()

    def _select_conversation(self, convo: Conversation) -> None:
        self.selected_conversation = convo
        self._render_conversation()
        self._select_row_by_id(convo.id)

    def _select_row_by_id(self, conversation_id: str) -> None:
        for row in self.listbox.get_children():
            if getattr(row, "conversation_id", None) == conversation_id:
                self.listbox.select_row(row)
                break

    def _render_conversation(self) -> None:
        self.textbuffer.set_text("")
        if not self.selected_conversation:
            return
        for message in self.selected_conversation.messages:
            self._append_message(message)
        self.textview.scroll_to_iter(self.textbuffer.get_end_iter(), 0.0, True, 0.0, 1.0)

    def _append_message(self, message: Message) -> None:
        display_name = (
            self.settings.user_name
            if message.role == "user"
            else self.settings.assistant_name
        )
        role_tag = (
            f"{message.role}_role" if message.role in {"user", "assistant"} else "role_base"
        )
        message_tag = (
            f"{message.role}_message"
            if message.role in {"user", "assistant"}
            else "assistant_message"
        )

        self.textbuffer.insert_with_tags_by_name(
            self.textbuffer.get_end_iter(), f"{display_name}\n", "role_base", role_tag
        )
        self.textbuffer.insert_with_tags_by_name(
            self.textbuffer.get_end_iter(), f"{message.timestamp}\n", "timestamp"
        )
        self._insert_formatted_content(message.content, message_tag)
        self._insert_images(getattr(message, "images", []))
        self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")

    def _insert_images(self, images: List[str]) -> None:
        for image_path in images:
            if not image_path:
                continue
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_path)
            except Exception:  # noqa: BLE001
                continue

            width = pixbuf.get_width()
            if width > 500:
                scaled_height = int(pixbuf.get_height() * (500 / width))
                pixbuf = pixbuf.scale_simple(500, scaled_height, GdkPixbuf.InterpType.BILINEAR)

            self.textbuffer.insert_pixbuf(self.textbuffer.get_end_iter(), pixbuf)
            self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")

    def _insert_formatted_content(self, text: str, message_tag: str) -> None:
        lines = text.splitlines() or [""]
        in_code_block = False
        for raw_line in lines:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), f"{line}\n", message_tag, "code"
                )
                continue

            if stripped in {"***", "---"}:
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), "\u2015" * 40 + "\n", "hr"
                )
                continue

            if line.startswith("### "):
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), line[4:] + "\n", message_tag, "heading3"
                )
                continue
            if line.startswith("## "):
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), line[3:] + "\n", message_tag, "heading2"
                )
                continue
            if line.startswith("# "):
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), line[2:] + "\n", message_tag, "heading1"
                )
                continue

            if stripped.startswith("- "):
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), "• ", message_tag, "bullet"
                )
                self._insert_inline_markup(stripped[2:] + "\n", message_tag)
                continue

            self._insert_inline_markup(line + "\n", message_tag)

    def _insert_inline_markup(self, text: str, message_tag: str) -> None:
        pattern = re.compile(r"(\${1,2})(.+?)\1")
        position = 0
        for match in pattern.finditer(text):
            start, end = match.span()
            if start > position:
                self._insert_basic_markup(text[position:start], message_tag)

            formula = match.group(2).strip()
            is_block = len(match.group(1)) == 2
            inserted = self._insert_latex(formula, is_block)
            if not inserted:
                self._insert_basic_markup(match.group(0), message_tag)
            position = end

        if position < len(text):
            self._insert_basic_markup(text[position:], message_tag)

    def _insert_basic_markup(self, text: str, message_tag: str) -> None:
        pattern = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*")
        position = 0
        for match in pattern.finditer(text):
            start, end = match.span()
            if start > position:
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), text[position:start], message_tag
                )

            bold_text = match.group(1)
            italic_text = match.group(2)
            if bold_text:
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), bold_text, message_tag, "bold"
                )
            elif italic_text:
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), italic_text, message_tag, "italic"
                )
            position = end

        if position < len(text):
            self.textbuffer.insert_with_tags_by_name(
                self.textbuffer.get_end_iter(), text[position:], message_tag
            )

    def _insert_latex(self, formula: str, block: bool) -> bool:
        pixbuf = self._render_latex_pixbuf(formula)
        if not pixbuf:
            return False

        if block:
            self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")
        self.textbuffer.insert_pixbuf(self.textbuffer.get_end_iter(), pixbuf)
        if block:
            self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")
        return True

    def _render_latex_pixbuf(self, formula: str) -> Optional[GdkPixbuf.Pixbuf]:
        mathtext_module = self._load_mathtext()
        if not mathtext_module:
            return None

        buffer = BytesIO()
        try:
            mathtext_module.math_to_image(f"${formula}$", buffer, dpi=160, format="png")
        except Exception:  # noqa: BLE001
            return None

        loader = GdkPixbuf.PixbufLoader.new_with_type("png")
        loader.write(buffer.getvalue())
        loader.close()
        return loader.get_pixbuf()

    def _load_mathtext(self):
        if self._mathtext is False:
            return None
        if self._mathtext:
            return self._mathtext
        if importlib.util.find_spec("matplotlib") is None:
            self._mathtext = False
            return None

        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import mathtext

        self._mathtext = mathtext
        return self._mathtext

    def _color_to_hex(self, color: Gdk.RGBA) -> str:
        return "#{:02x}{:02x}{:02x}".format(
            int(color.red * 255), int(color.green * 255), int(color.blue * 255)
        )

    def on_open_settings(self, _button: Gtk.Button) -> None:
        dialog = Gtk.Dialog(title="Settings", transient_for=self, modal=True)
        dialog.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        dialog.add_button(Gtk.STOCK_SAVE, Gtk.ResponseType.OK)

        content = dialog.get_content_area()
        grid = Gtk.Grid(column_spacing=8, row_spacing=8, margin=12)
        content.add(grid)

        font_label = Gtk.Label(label="Font size", xalign=0)
        font_spin = Gtk.SpinButton.new_with_range(8, 32, 1)
        font_spin.set_value(self.settings.font_size)
        grid.attach(font_label, 0, 0, 1, 1)
        grid.attach(font_spin, 1, 0, 1, 1)

        user_color_label = Gtk.Label(label="User color", xalign=0)
        user_color_button = Gtk.ColorButton()
        user_rgba = Gdk.RGBA()
        user_rgba.parse(self.settings.user_color)
        user_color_button.set_rgba(user_rgba)
        grid.attach(user_color_label, 0, 1, 1, 1)
        grid.attach(user_color_button, 1, 1, 1, 1)

        assistant_color_label = Gtk.Label(label="Assistant color", xalign=0)
        assistant_color_button = Gtk.ColorButton()
        assistant_rgba = Gdk.RGBA()
        assistant_rgba.parse(self.settings.assistant_color)
        assistant_color_button.set_rgba(assistant_rgba)
        grid.attach(assistant_color_label, 0, 2, 1, 1)
        grid.attach(assistant_color_button, 1, 2, 1, 1)

        user_name_label = Gtk.Label(label="User name", xalign=0)
        user_name_entry = Gtk.Entry()
        user_name_entry.set_text(self.settings.user_name)
        grid.attach(user_name_label, 0, 3, 1, 1)
        grid.attach(user_name_entry, 1, 3, 1, 1)

        assistant_name_label = Gtk.Label(label="Assistant name", xalign=0)
        assistant_name_entry = Gtk.Entry()
        assistant_name_entry.set_text(self.settings.assistant_name)
        grid.attach(assistant_name_label, 0, 4, 1, 1)
        grid.attach(assistant_name_entry, 1, 4, 1, 1)

        dialog.show_all()
        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            self.settings.font_size = font_spin.get_value_as_int()
            self.settings.user_color = self._color_to_hex(user_color_button.get_rgba())
            self.settings.assistant_color = self._color_to_hex(
                assistant_color_button.get_rgba()
            )
            self.settings.user_name = user_name_entry.get_text() or "User"
            self.settings.assistant_name = assistant_name_entry.get_text() or "Assistant"
            self.settings_store.save()
            self._apply_settings()
            self._render_conversation()

        dialog.destroy()

    def on_new_chat(self, _button: Gtk.Button) -> None:
        title = "New conversation"
        model = DEFAULT_MODELS[0][0]
        convo = self.store.create(title=title, model=model)
        self._refresh_sidebar()
        self._select_conversation(convo)

    def on_delete_chat(self, _button: Gtk.Button) -> None:
        if not self.selected_conversation:
            return
        convo_id = self.selected_conversation.id
        self.store.delete(convo_id)
        self.selected_conversation = None
        self._refresh_sidebar()
        if self.store.conversations:
            self._select_conversation(self.store.conversations[0])
        else:
            self.textbuffer.set_text("")

    def on_conversation_selected(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow) -> None:
        if not row:
            return
        convo_id = getattr(row, "conversation_id", None)
        convo = self.store.get(convo_id)
        if convo:
            self.selected_conversation = convo
            self._render_conversation()

    def on_send(self, _widget: Gtk.Widget) -> None:
        text = self.entry.get_text().strip()
        if not text or not self.selected_conversation:
            return

        model_index = self.model_combo.get_active()
        model_value = DEFAULT_MODELS[model_index][0]
        self.selected_conversation.model = model_value

        user_msg = Message(role="user", content=text)
        self.selected_conversation.messages.append(user_msg)
        self.store.save()
        self._render_conversation()
        self.entry.set_text("")

        conversation_id = self.selected_conversation.id
        threading.Thread(
            target=self._respond_in_thread, args=(conversation_id,), daemon=True
        ).start()

    def _respond_in_thread(self, conversation_id: str) -> None:
        convo = self.store.get(conversation_id)
        if not convo:
            return

        success, reply_content, images = self.model_client.generate_reply(convo)
        GLib.idle_add(
            self._apply_response, conversation_id, success, reply_content, images
        )

    def _apply_response(
        self, conversation_id: str, success: bool, reply: str, images: List[str]
    ) -> bool:
        convo = self.store.get(conversation_id)
        if not convo:
            return False

        content = reply if success else f"(Model error) {reply}"
        assistant_msg = Message(role="assistant", content=content, images=images)
        convo.messages.append(assistant_msg)

        if convo.title == "New conversation" and convo.messages:
            first_user_message = next((m for m in convo.messages if m.role == "user"), None)
            if first_user_message:
                convo.title = first_user_message.content[:30] + ("…" if len(first_user_message.content) > 30 else "")

        self.store.save()
        if self.selected_conversation and self.selected_conversation.id == convo.id:
            self._refresh_sidebar()
            self._render_conversation()
        return False


class GeminiApplication(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id="dev.gemini.gtk")
        self.store = ConversationStore(CONVERSATIONS_FILE)
        self.settings_store = SettingsStore(SETTINGS_FILE)

    def do_activate(self) -> None:
        win = ChatWindow(self, self.store, self.settings_store)
        win.show_all()


def main() -> None:
    app = GeminiApplication()
    app.run()


if __name__ == "__main__":
    main()
