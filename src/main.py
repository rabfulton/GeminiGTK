import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, Pango

from google import genai
from google.genai import types


DATA_DIR = Path.home() / ".gemini_gtk"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONVERSATIONS_FILE = DATA_DIR / "conversations.json"
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
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Conversation:
    id: str
    title: str
    model: str
    messages: List[Message] = field(default_factory=list)


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
                    Message(**msg)
                    for msg in item.get("messages", [])
                    if msg.get("content")
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
                    {"role": m.role, "content": m.content, "timestamp": m.timestamp}
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
        self.configuration_error: Optional[str] = None

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
        if not api_key:
            self.configuration_error = (
                "Set GEMINI_API_KEY or GOOGLE_GENAI_API_KEY to enable real model calls."
            )
            return

        self.client = genai.Client(api_key=api_key)

    def generate_reply(self, conversation: Conversation) -> Tuple[bool, str]:
        if not self.client:
            return False, self.configuration_error or "API client is not configured."

        contents = [
            types.Content(
                role="user" if message.role == "user" else "model",
                parts=[types.Part.from_text(message.content)],
            )
            for message in conversation.messages
            if message.content
        ]
        if not contents:
            return False, "Conversation has no messages to send."

        try:
            response = self.client.models.generate_content(
                model=conversation.model,
                contents=contents,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"API error: {exc}"

        text = getattr(response, "text", "") or self._extract_candidate_text(response)
        if not text:
            return False, "Received empty response from the model."
        return True, text.strip()

    @staticmethod
    def _extract_candidate_text(response: object) -> str:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for part in parts:
                text = getattr(part, "text", "")
                if text:
                    return text
        return ""


class ChatWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, store: ConversationStore):
        super().__init__(application=app)
        self.set_default_size(900, 600)
        self.set_title("Gemini GTK")

        self.store = store
        self.selected_conversation: Optional[Conversation] = None
        self.model_client = ModelClient()

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
        header.pack_start(new_button, True, True, 0)
        header.pack_start(delete_button, True, True, 0)
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
        self.textbuffer.create_tag("role", weight=Pango.Weight.BOLD, scale=1.05)
        self.textbuffer.create_tag("timestamp", foreground="#777777", scale=0.9)
        self.textbuffer.create_tag("message", pixels_above_lines=2, pixels_below_lines=6)

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
        cursor = self.textbuffer.get_start_iter()
        for message in self.selected_conversation.messages:
            self._append_message(cursor, message)
            cursor = self.textbuffer.get_end_iter()
        self.textview.scroll_to_iter(self.textbuffer.get_end_iter(), 0.0, True, 0.0, 1.0)

    def _append_message(self, iter_start: Gtk.TextIter, message: Message) -> None:
        self.textbuffer.insert_with_tags_by_name(iter_start, f"{message.role.capitalize()}\n", "role")
        self.textbuffer.insert_with_tags_by_name(iter_start, f"{message.timestamp}\n", "timestamp")
        self.textbuffer.insert_with_tags_by_name(iter_start, f"{message.content}\n\n", "message")

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

        success, reply_content = self.model_client.generate_reply(convo)
        GLib.idle_add(self._apply_response, conversation_id, success, reply_content)

    def _apply_response(self, conversation_id: str, success: bool, reply: str) -> bool:
        convo = self.store.get(conversation_id)
        if not convo:
            return False

        content = reply if success else f"(Model error) {reply}"
        assistant_msg = Message(role="assistant", content=content)
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

    def do_activate(self) -> None:
        win = ChatWindow(self, self.store)
        win.show_all()


def main() -> None:
    app = GeminiApplication()
    app.run()


if __name__ == "__main__":
    main()
