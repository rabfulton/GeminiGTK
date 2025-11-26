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
from typing import List, Optional, Tuple, TYPE_CHECKING

import gi

if TYPE_CHECKING:
    from PIL import Image

gi.require_version("Gtk", "3.0")
gi.require_version('GdkPixbuf', '2.0')
gi.require_version('Gdk', '3.0')
gi.require_version('GtkSource', '3.0')
from gi.repository import GLib, Gdk, GdkPixbuf, Gtk, GtkSource, Pango

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
    window_width: int = 900
    window_height: int = 600
    window_x: Optional[int] = None
    window_y: Optional[int] = None
    image_resolution: str = "2K"  # Default to 2K for Gemini 3
    image_aspect_ratio: str = "1:1"


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
            window_width=int(data.get("window_width", self.settings.window_width)),
            window_height=int(data.get("window_height", self.settings.window_height)),
            window_x=data.get("window_x", self.settings.window_x),
            window_y=data.get("window_y", self.settings.window_y),
            image_resolution=data.get("image_resolution", self.settings.image_resolution),
            image_aspect_ratio=data.get("image_aspect_ratio", self.settings.image_aspect_ratio),
        )

    def save(self) -> None:
        payload = {
            "font_size": self.settings.font_size,
            "user_color": self.settings.user_color,
            "assistant_color": self.settings.assistant_color,
            "user_name": self.settings.user_name,
            "assistant_name": self.settings.assistant_name,
            "window_width": self.settings.window_width,
            "window_height": self.settings.window_height,
            "window_x": self.settings.window_x,
            "window_y": self.settings.window_y,
            "image_resolution": self.settings.image_resolution,
            "image_aspect_ratio": self.settings.image_aspect_ratio,
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
    def __init__(self, settings: Settings) -> None:
        self.client = None
        self.types = None
        self.settings = settings
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

        contents = []
        for message in conversation.messages:
            if message.content or message.images:
                parts = []
                if message.content:
                    parts.append(self.types.Part(text=message.content))

                # Handle user-provided images
                for image_path in message.images:
                    if image_path and os.path.exists(image_path):
                        try:
                            from PIL import Image
                            pil_image = Image.open(image_path)
                            # Convert to RGB if necessary
                            if pil_image.mode not in ('RGB', 'RGBA'):
                                pil_image = pil_image.convert('RGB')
                            parts.append(self.types.Part(inline_data=self._image_to_inline_data(pil_image)))
                        except Exception as exc:  # noqa: BLE001
                            # If image loading fails, continue without it
                            continue

                if parts:
                    contents.append(self.types.Content(
                        role="user" if message.role == "user" else "model",
                        parts=parts,
                    ))

        if not contents:
            return False, "Conversation has no messages to send.", []

        # Set generation config for image models
        config = None
        is_image_model = conversation.model in ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]
        if is_image_model:
            # Handle different models with their supported resolutions
            if conversation.model == "gemini-2.5-flash-image":
                # Gemini 2.5 Flash Image only supports 1K resolution
                image_size = "1K"
                aspect_ratio = self.settings.image_aspect_ratio
            else:
                # Gemini 3 Pro Image Preview supports 1K, 2K, and 4K resolutions
                image_size = self.settings.image_resolution  # "1K", "2K", or "4K"
                aspect_ratio = self.settings.image_aspect_ratio

            image_config = self.types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size
            )

            config = self.types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=image_config
            )

        try:
            response = self.client.models.generate_content(
                model=conversation.model,
                contents=contents,
                config=config,
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

    def _image_to_inline_data(self, pil_image: "Image") -> object:
        """Convert a PIL Image to inline data format for the API."""
        from io import BytesIO

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        return self.types.Blob(
            mime_type="image/png",
            data=image_bytes,
        )

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
        self.set_title("Gemini GTK")

        self.store = store
        self.settings_store = settings_store
        self.settings = settings_store.settings
        self.selected_conversation: Optional[Conversation] = None
        self.model_client = ModelClient(self.settings)
        self._mathtext = None
        self._font_manager = None
        self.pending_images: List[str] = []
        self._resize_connected = False
        self._resize_idle_id: Optional[int] = None
        self._last_textview_width: Optional[int] = None
        self._textview_css_provider: Optional[Gtk.CssProvider] = None

        self._restore_window_geometry()
        self.connect("delete-event", self._on_window_delete_event)

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

    def _restore_window_geometry(self) -> None:
        """Resize/move the window using the last stored geometry."""
        width = max(400, int(getattr(self.settings, "window_width", 900) or 900))
        height = max(300, int(getattr(self.settings, "window_height", 600) or 600))
        self.resize(width, height)
        if (
            getattr(self.settings, "window_x", None) is not None
            and getattr(self.settings, "window_y", None) is not None
        ):
            self.move(int(self.settings.window_x), int(self.settings.window_y))

    def _persist_window_geometry(self) -> None:
        """Capture the current window geometry and save it to disk."""
        width, height = self.get_size()
        self.settings.window_width = width
        self.settings.window_height = height
        x, y = self.get_position()
        self.settings.window_x = x
        self.settings.window_y = y
        self.settings_store.save()

    def _on_window_delete_event(self, *_args) -> bool:
        self._persist_window_geometry()
        return False

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
        self.textview.get_style_context().add_class("chat-textview")
        self._ensure_textview_css()
        self._apply_textview_margins()

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

        self.image_count_label = Gtk.Label()
        self.image_count_label.hide()
        box.pack_start(self.image_count_label, False, False, 0)

        image_button = Gtk.Button(label="📎")
        image_button.set_tooltip_text("Attach image")
        image_button.connect("clicked", self.on_attach_image)
        box.pack_start(image_button, False, False, 0)

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
        self._ensure_textview_css()
        self._apply_textview_margins()
        self._update_textview_font()

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

    def _ensure_textview_css(self) -> None:
        if not self._textview_css_provider:
            self._textview_css_provider = Gtk.CssProvider()
        context = self.textview.get_style_context()
        context.add_provider(self._textview_css_provider, Gtk.STYLE_PROVIDER_PRIORITY_USER)
        self._update_textview_font()

    def _apply_textview_margins(self) -> None:
        self.textview.set_left_margin(12)
        self.textview.set_right_margin(12)
        self.textview.set_top_margin(8)
        self.textview.set_bottom_margin(8)
        self._apply_margins_to_embedded_widgets()

    def _apply_margins_to_embedded_widgets(self) -> None:
        left = self.textview.get_left_margin()
        right = self.textview.get_right_margin()
        top = self.textview.get_top_margin()
        bottom = self.textview.get_bottom_margin()

        for child in self.textview.get_children():
            if isinstance(child, Gtk.ScrolledWindow):
                child.set_margin_start(left)
                child.set_margin_end(right)
                child.set_margin_top(top)
                child.set_margin_bottom(bottom)

    def _update_textview_font(self) -> None:
        if not self._textview_css_provider:
            return
        font_size = max(8, self.settings.font_size)
        css = f".chat-textview {{ font-size: {font_size}pt; }}"
        try:
            self._textview_css_provider.load_from_data(css.encode("utf-8"))
        except Exception:  # noqa: BLE001
            pass

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
        self._apply_textview_margins()
        for child in list(self.textview.get_children()):
            self.textview.remove(child)
        self.textbuffer.set_text("")
        if not self.selected_conversation:
            return
        for message in self.selected_conversation.messages:
            self._append_message(message)
        self._apply_margins_to_embedded_widgets()
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
        # Store image info for potential redraw
        if not hasattr(self, "_images_to_insert"):
            self._images_to_insert = []
        self._images_to_insert = images.copy()  # always latest set

        for image_path in images:
            if not image_path:
                continue
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_path)
            except Exception:  # noqa: BLE001
                continue

            width = pixbuf.get_width()
            textview_width = self.textview.get_allocated_width()
            max_width = max(100, textview_width - 40) if textview_width > 0 else 500
            if width > max_width:
                scaled_height = int(pixbuf.get_height() * (max_width / width))
                pixbuf = pixbuf.scale_simple(max_width, scaled_height, GdkPixbuf.InterpType.BILINEAR)

            self.textbuffer.insert_pixbuf(self.textbuffer.get_end_iter(), pixbuf)
            self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")

        self._ensure_resize_handler()

    def _ensure_resize_handler(self) -> None:
        if self._resize_connected:
            return

        self.textview.connect("size-allocate", self._on_textview_size_allocate)
        self._resize_connected = True

    def _on_textview_size_allocate(self, _widget: Gtk.Widget, allocation: Gdk.Rectangle) -> None:
        new_width = allocation.width
        if new_width <= 0:
            return
        if new_width == self._last_textview_width:
            return

        self._last_textview_width = new_width
        if self._resize_idle_id is not None:
            GLib.source_remove(self._resize_idle_id)
        # Schedule a redraw once resizing settles to avoid thrashing
        self._resize_idle_id = GLib.timeout_add(80, self._redraw_conversation_after_resize)

    def _redraw_conversation_after_resize(self) -> bool:
        self._resize_idle_id = None
        if self.selected_conversation:
            self._render_conversation()
        return False

    def _insert_formatted_content(self, text: str, message_tag: str) -> None:
        lines = text.splitlines() or [""]
        in_code_block = False
        code_lines: List[str] = []
        code_language: Optional[str] = None
        index = 0
        while index < len(lines):
            raw_line = lines[index]
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("```"):
                if in_code_block:
                    self._insert_code_block("\n".join(code_lines), code_language)
                    code_lines = []
                    code_language = None
                else:
                    code_language = stripped[3:].strip() or None
                in_code_block = not in_code_block
                index += 1
                continue

            if in_code_block:
                code_lines.append(line)
                index += 1
                continue

            table_parse = self._maybe_parse_table(lines, index)
            if table_parse:
                rows, aligns, next_index = table_parse
                self._insert_table(rows, aligns, message_tag)
                index = next_index
                continue

            if stripped in {"***", "---"}:
                self._insert_horizontal_rule()
                index += 1
                continue

            if line.startswith("### "):
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), line[4:] + "\n", message_tag, "heading3"
                )
                index += 1
                continue
            if line.startswith("## "):
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), line[3:] + "\n", message_tag, "heading2"
                )
                index += 1
                continue
            if line.startswith("# "):
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), line[2:] + "\n", message_tag, "heading1"
                )
                index += 1
                continue

            bullet_match = re.match(r"^([-*])\s+(.*)", stripped)
            if bullet_match:
                self.textbuffer.insert_with_tags_by_name(
                    self.textbuffer.get_end_iter(), "• ", message_tag, "bullet"
                )
                self._insert_inline_markup(bullet_match.group(2) + "\n", message_tag)
                index += 1
                continue

            self._insert_inline_markup(line + "\n", message_tag)
            index += 1

        if in_code_block and code_lines:
            self._insert_code_block("\n".join(code_lines), code_language)

    def _insert_code_block(self, code: str, language_hint: Optional[str]) -> None:
        buffer = GtkSource.Buffer()
        buffer.set_text(code)

        language_manager = GtkSource.LanguageManager.get_default()
        language = None
        if language_hint:
            language = language_manager.get_language(language_hint)
            if not language:
                language = language_manager.guess_language(f"file.{language_hint}", None)
        if not language:
            language = language_manager.guess_language(None, "text/plain")
        if language:
            buffer.set_language(language)
        buffer.set_highlight_syntax(True)

        source_view = GtkSource.View.new_with_buffer(buffer)
        source_view.set_editable(False)
        source_view.set_cursor_visible(False)
        source_view.set_wrap_mode(Gtk.WrapMode.NONE)
        source_view.set_monospace(True)
        source_view.set_left_margin(8)
        source_view.set_right_margin(8)

        line_count = max(1, code.count("\n") + 1)
        layout = source_view.create_pango_layout("M")
        _, line_height = layout.get_pixel_size()
        desired_height = line_height * line_count + 16

        source_view.set_size_request(-1, desired_height)
        source_view.set_hexpand(True)
        source_view.set_halign(Gtk.Align.FILL)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
        scrolled.set_propagate_natural_width(True)
        scrolled.set_propagate_natural_height(True)
        scrolled.set_margin_start(self.textview.get_left_margin())
        scrolled.set_margin_end(self.textview.get_right_margin())
        scrolled.set_margin_top(self.textview.get_top_margin())
        scrolled.set_margin_bottom(self.textview.get_bottom_margin())
        scrolled.set_min_content_height(desired_height)
        scrolled.set_size_request(-1, desired_height)
        scrolled.set_hexpand(True)
        scrolled.set_halign(Gtk.Align.FILL)
        textview_width = self.textview.get_allocated_width()
        min_width = textview_width if textview_width > 0 else 600
        scrolled.set_min_content_width(min_width)
        scrolled.add(source_view)
        scrolled.show_all()

        self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")
        anchor = self.textbuffer.create_child_anchor(self.textbuffer.get_end_iter())
        self.textview.add_child_at_anchor(scrolled, anchor)
        self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")

    def _insert_inline_markup(
        self,
        text: str,
        message_tag: str,
        buffer: Optional[Gtk.TextBuffer] = None,
        base_tags: Optional[List[str]] = None,
    ) -> None:
        target_buffer = buffer or self.textbuffer
        text = self._strip_emphasis_from_math(text)
        emphasis_pattern = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*")
        position = 0
        effective_tags = base_tags or [message_tag]
        for match in emphasis_pattern.finditer(text):
            if match.start() > position:
                self._insert_text_with_math(
                    text[position:match.start()], effective_tags, target_buffer
                )

            content = match.group(1) or match.group(2)
            tags = effective_tags + (["bold"] if match.group(1) else ["italic"])
            self._insert_text_with_math(content, tags, target_buffer)
            position = match.end()

        if position < len(text):
            self._insert_text_with_math(text[position:], effective_tags, target_buffer)

    def _strip_emphasis_from_math(self, text: str) -> str:
        pattern = re.compile(r"(\*\*|\*)(\${1,2})(.+?)\2\1")
        return pattern.sub(lambda match: f"{match.group(2)}{match.group(3)}{match.group(2)}", text)

    def _insert_text_with_math(
        self,
        text: str,
        tags: List[str],
        buffer: Optional[Gtk.TextBuffer] = None,
    ) -> None:
        target_buffer = buffer or self.textbuffer
        pattern = re.compile(r"(\${1,2})(.+?)\1")
        position = 0
        for match in pattern.finditer(text):
            start, end = match.span()
            if start > position:
                target_buffer.insert_with_tags_by_name(
                    target_buffer.get_end_iter(), text[position:start], *tags
                )

            formula = match.group(2).strip()
            is_block = len(match.group(1)) == 2
            inserted = self._insert_latex(formula, is_block, target_buffer)
            if not inserted:
                target_buffer.insert_with_tags_by_name(
                    target_buffer.get_end_iter(), match.group(0), *tags
                )
            position = end

        if position < len(text):
            target_buffer.insert_with_tags_by_name(
                target_buffer.get_end_iter(), text[position:], *tags
            )

    def _insert_latex(
        self,
        formula: str,
        block: bool,
        buffer: Optional[Gtk.TextBuffer] = None,
    ) -> bool:
        target_buffer = buffer or self.textbuffer
        pixbuf = self._render_latex_pixbuf(formula)
        if not pixbuf:
            return False

        if block:
            target_buffer.insert(target_buffer.get_end_iter(), "\n")
        target_buffer.insert_pixbuf(target_buffer.get_end_iter(), pixbuf)
        if block:
            target_buffer.insert(target_buffer.get_end_iter(), "\n")
        return True

    def _maybe_parse_table(
        self, lines: List[str], start_index: int
    ) -> Optional[Tuple[List[List[str]], List[str], int]]:
        header_line = lines[start_index].strip()
        if "|" not in header_line:
            return None
        if start_index + 1 >= len(lines):
            return None
        separator_line = lines[start_index + 1].strip()
        header_cells = self._split_table_row_line(header_line)
        if not header_cells:
            return None
        aligns = self._parse_alignment_line(separator_line, len(header_cells))
        if not aligns:
            return None

        rows: List[List[str]] = [header_cells]
        index = start_index + 2
        while index < len(lines):
            current = lines[index]
            if not current.strip():
                break
            if "|" not in current:
                break
            rows.append(self._split_table_row_line(current))
            index += 1

        if len(rows) < 2:
            return None
        max_cols = max(len(row) for row in rows)
        for row in rows:
            if len(row) < max_cols:
                row.extend([""] * (max_cols - len(row)))
        if len(aligns) < max_cols:
            aligns.extend(["left"] * (max_cols - len(aligns)))
        return rows, aligns[:max_cols], index

    def _split_table_row_line(self, line: str) -> List[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]

        cells: List[str] = []
        current: List[str] = []
        index = 0
        length = len(stripped)
        while index < length:
            char = stripped[index]
            next_char = stripped[index + 1] if index + 1 < length else ""
            if char == "\\" and next_char in {"|", "\\"}:
                current.append(next_char)
                index += 2
                continue
            if char == "|":
                cells.append("".join(current).strip())
                current = []
            else:
                current.append(char)
            index += 1
        cells.append("".join(current).strip())
        return cells

    def _parse_alignment_line(self, line: str, expected_cols: int) -> Optional[List[str]]:
        stripped = line.strip()
        if not stripped:
            return None
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        segments = [segment.strip() for segment in stripped.split("|")]
        aligns: List[str] = []
        alignment_pattern = re.compile(r"^:?-{3,}:?$")
        for segment in segments:
            if not alignment_pattern.match(segment):
                return None
            if segment.startswith(":") and segment.endswith(":"):
                aligns.append("center")
            elif segment.endswith(":"):
                aligns.append("right")
            else:
                aligns.append("left")
        if len(aligns) < expected_cols:
            aligns.extend(["left"] * (expected_cols - len(aligns)))
        return aligns[:expected_cols]

    def _insert_table(
        self,
        rows: List[List[str]],
        aligns: List[str],
        message_tag: str,
    ) -> None:
        grid = Gtk.Grid()
        grid.set_row_spacing(4)
        grid.set_column_spacing(8)
        grid.set_margin_top(6)
        grid.set_margin_bottom(6)
        grid.set_margin_start(6)
        grid.set_margin_end(6)
        grid.set_column_homogeneous(False)
        grid.set_hexpand(True)
        grid.set_vexpand(False)

        textview_width = self.textview.get_allocated_width()
        table_width = max(250, (textview_width - 40) if textview_width > 0 else 700)
        column_count = max(len(row) for row in rows)
        column_width = max(80, (table_width - (grid.get_column_spacing() * (column_count - 1))) // max(1, column_count))
        size_groups = [
            Gtk.SizeGroup(mode=Gtk.SizeGroupMode.HORIZONTAL) for _ in range(column_count)
        ]

        for row_index, row in enumerate(rows):
            is_header = row_index == 0
            for col_index in range(column_count):
                cell_text = row[col_index] if col_index < len(row) else ""
                alignment = aligns[col_index] if col_index < len(aligns) else "left"
                cell_view = self._create_table_cell_view(
                    cell_text,
                    message_tag,
                    alignment,
                    is_header,
                    column_width,
                )
                grid.attach(cell_view, col_index, row_index, 1, 1)
                size_groups[col_index].add_widget(cell_view)

        frame = Gtk.Frame()
        frame.set_shadow_type(Gtk.ShadowType.IN)
        frame.set_hexpand(True)
        frame.set_vexpand(False)
        frame.set_size_request(table_width, -1)
        frame.add(grid)
        frame.show_all()

        self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")
        anchor = self.textbuffer.create_child_anchor(self.textbuffer.get_end_iter())
        self.textview.add_child_at_anchor(frame, anchor)
        self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")

    def _insert_horizontal_rule(self) -> None:
        self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")

        available_width = self.textview.get_allocated_width()
        if available_width <= 0:
            available_width = max(400, self.get_allocated_width() - 60)
        width = max(80, available_width - 24)

        pixbuf = self._create_horizontal_rule_pixbuf(width, 1)
        if pixbuf:
            self.textbuffer.insert_pixbuf(self.textbuffer.get_end_iter(), pixbuf)
        self.textbuffer.insert(self.textbuffer.get_end_iter(), "\n")

    def _create_table_cell_view(
        self,
        text: str,
        message_tag: str,
        alignment: str,
        is_header: bool,
        preferred_width: int,
    ) -> Gtk.Widget:
        textview = Gtk.TextView()
        textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        textview.set_editable(False)
        textview.set_cursor_visible(False)
        textview.set_left_margin(4)
        textview.set_right_margin(4)
        textview.set_top_margin(2)
        textview.set_bottom_margin(2)
        textview.set_hexpand(True)
        textview.set_vexpand(False)
        if preferred_width > 0:
            textview.set_size_request(preferred_width, -1)
        if alignment == "center":
            textview.set_justification(Gtk.Justification.CENTER)
        elif alignment == "right":
            textview.set_justification(Gtk.Justification.RIGHT)
        else:
            textview.set_justification(Gtk.Justification.LEFT)

        buffer = textview.get_buffer()
        self._ensure_buffer_tags(buffer, message_tag)
        base_tags = [message_tag, "bold"] if is_header else [message_tag]
        self._insert_inline_markup(text or "", message_tag, buffer, base_tags=base_tags)
        return textview

    def _ensure_buffer_tags(self, buffer: Gtk.TextBuffer, message_tag: str) -> None:
        tag_table = buffer.get_tag_table()

        def ensure_tag(name: str, **properties) -> None:
            tag = tag_table.lookup(name)
            if not tag:
                tag = buffer.create_tag(name)
            for key, value in properties.items():
                tag.set_property(key.replace("_", "-"), value)

        ensure_tag("bold", weight=Pango.Weight.BOLD)
        ensure_tag("italic", style=Pango.Style.ITALIC)

        color_map = {
            "user_message": self.settings.user_color,
            "assistant_message": self.settings.assistant_color,
        }
        color = color_map.get(message_tag, self.settings.assistant_color)
        ensure_tag(message_tag, foreground=color)

    def _create_horizontal_rule_pixbuf(self, width: int, height: int) -> Optional[GdkPixbuf.Pixbuf]:
        width = max(1, int(width))
        height = max(1, int(height))
        try:
            pixbuf = GdkPixbuf.Pixbuf.new(
                colorspace=GdkPixbuf.Colorspace.RGB,
                has_alpha=True,
                bits_per_sample=8,
                width=width,
                height=height,
            )
        except Exception:  # noqa: BLE001
            return None

        color = Gdk.RGBA()
        if not color.parse(self.settings.assistant_color):
            color.parse("#1b5e20")

        red = min(255, max(0, int(color.red * 255)))
        green = min(255, max(0, int(color.green * 255)))
        blue = min(255, max(0, int(color.blue * 255)))
        alpha = min(255, max(0, int(color.alpha * 255))) or 255

        pixel_value = (red << 24) | (green << 16) | (blue << 8) | alpha
        pixbuf.fill(pixel_value)
        return pixbuf

    def _render_latex_pixbuf(self, formula: str) -> Optional[GdkPixbuf.Pixbuf]:
        mathtext_module = self._load_mathtext()
        if not mathtext_module or not self._font_manager:
            return None

        buffer = BytesIO()
        try:
            font_size = getattr(self.settings, "font_size", 12)
            tex_font = self._font_manager.FontProperties(
                size=font_size, family="serif", math_fontfamily="cm"
            )
            screen = self.get_screen()
            dpi = screen.get_resolution() if screen else 96.0
            if not dpi or dpi <= 0:
                dpi = 96.0
            scale_factor = getattr(self, "get_scale_factor", lambda: 1)() or 1
            effective_dpi = dpi * max(1, scale_factor)
            mathtext_module.math_to_image(
                f"${formula}$", buffer, prop=tex_font, dpi=effective_dpi, format="png"
            )
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

        matplotlib.rcParams['savefig.transparent'] = True
        matplotlib.rcParams['text.color'] = self.settings.assistant_color
        matplotlib.use("Agg")
        from matplotlib import mathtext, font_manager

        self._mathtext = mathtext
        self._font_manager = font_manager
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

        image_resolution_label = Gtk.Label(label="Image resolution", xalign=0)
        image_resolution_combo = Gtk.ComboBoxText()
        for resolution in ["1K", "2K", "4K"]:
            image_resolution_combo.append_text(resolution)
        try:
            image_resolution_combo.set_active(["1K", "2K", "4K"].index(self.settings.image_resolution))
        except ValueError:
            image_resolution_combo.set_active(1)  # Default to 2K
        grid.attach(image_resolution_label, 0, 5, 1, 1)
        grid.attach(image_resolution_combo, 1, 5, 1, 1)

        image_aspect_ratio_label = Gtk.Label(label="Image aspect ratio", xalign=0)
        image_aspect_ratio_combo = Gtk.ComboBoxText()
        aspect_ratios = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
        for ratio in aspect_ratios:
            image_aspect_ratio_combo.append_text(ratio)
        try:
            image_aspect_ratio_combo.set_active(aspect_ratios.index(self.settings.image_aspect_ratio))
        except ValueError:
            image_aspect_ratio_combo.set_active(0)  # Default to 1:1
        grid.attach(image_aspect_ratio_label, 0, 6, 1, 1)
        grid.attach(image_aspect_ratio_combo, 1, 6, 1, 1)

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
            self.settings.image_resolution = image_resolution_combo.get_active_text() or "2K"
            self.settings.image_aspect_ratio = image_aspect_ratio_combo.get_active_text() or "1:1"
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

    def on_attach_image(self, _button: Gtk.Button) -> None:
        dialog = Gtk.FileChooserDialog(
            title="Select Image",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK,
        )

        # Add image filters
        filter_images = Gtk.FileFilter()
        filter_images.set_name("Image files")
        filter_images.add_mime_type("image/*")
        dialog.add_filter(filter_images)

        filter_all = Gtk.FileFilter()
        filter_all.set_name("All files")
        filter_all.add_pattern("*")
        dialog.add_filter(filter_all)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
            if filename:
                self.pending_images.append(filename)
                self._update_input_bar_display()

        dialog.destroy()

    def _update_input_bar_display(self) -> None:
        if self.pending_images:
            self.image_count_label.set_text(f"({len(self.pending_images)} image{'s' if len(self.pending_images) != 1 else ''})")
            self.image_count_label.show()
        else:
            self.image_count_label.hide()

    def on_send(self, _widget: Gtk.Widget) -> None:
        text = self.entry.get_text().strip()
        if not text or not self.selected_conversation:
            return

        model_index = self.model_combo.get_active()
        model_value = DEFAULT_MODELS[model_index][0]
        self.selected_conversation.model = model_value

        user_msg = Message(role="user", content=text, images=self.pending_images.copy())
        self.selected_conversation.messages.append(user_msg)
        self.store.save()
        self._render_conversation()
        self.entry.set_text("")
        self.pending_images.clear()
        self._update_input_bar_display()

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
