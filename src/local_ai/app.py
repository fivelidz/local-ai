"""
Local AI - tmux-style Terminal Chat Interface for Ollama
Features: Collapsible panels, shortcuts guide, mouse support, animations
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Static, Input, Button, 
    ListView, ListItem, Label, Select, LoadingIndicator,
    TabbedContent, TabPane, OptionList, Collapsible
)
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual import on, work
from textual.worker import Worker, get_current_worker
from datetime import datetime
from pathlib import Path
import json
import httpx
import asyncio

# Chat storage directory
CHATS_DIR = Path.home() / "Models" / "chats"
CHATS_DIR.mkdir(parents=True, exist_ok=True)

# Loading animations
THINKING_ANIMATIONS = [
    "🤔 Thinking.",
    "🤔 Thinking..",
    "🤔 Thinking...",
    "🧠 Processing.",
    "🧠 Processing..",
    "🧠 Processing...",
    "✨ Generating.",
    "✨ Generating..",
    "✨ Generating...",
]


class ShortcutsScreen(ModalScreen):
    """Modal screen showing keyboard shortcuts"""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("enter", "dismiss", "Close"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("""[bold cyan]
╔═══════════════════════════════════════════════════════════════╗
║                     LOCAL AI - SHORTCUTS                       ║
╚═══════════════════════════════════════════════════════════════╝[/]

[bold green]Navigation[/]
  [yellow]Tab[/]           Cycle between panels
  [yellow]Ctrl+1[/]        Focus Sessions panel
  [yellow]Ctrl+2[/]        Focus Models panel
  [yellow]Ctrl+3[/]        Focus Chat input
  [yellow]Escape[/]        Focus Chat input

[bold green]Actions[/]
  [yellow]Ctrl+N[/]        New Chat
  [yellow]Ctrl+S[/]        Toggle Sessions panel
  [yellow]Ctrl+M[/]        Toggle Models panel
  [yellow]Enter[/]         Send message
  [yellow]Ctrl+Q[/]        Quit

[bold green]Mouse Controls[/]
  [yellow]Click[/]         Select model or session
  [yellow]Scroll[/]        Scroll through messages
  [yellow]Double-click[/]  Quick select and focus

[bold green]Model Categories[/]
  [yellow]⚡[/] Fast       Low latency, quick responses
  [yellow]🎯[/] Balanced   Good quality/speed balance
  [yellow]💻[/] Coding     Optimized for code
  [yellow]🚀[/] Large      Best quality (CPU mode)

[bold green]Tips[/]
  • Click [cyan]➕ New Chat[/] to start a conversation
  • Select a model before chatting
  • Chats are auto-saved to ~/Models/chats/
  • Use CPU models (🚀) for complex tasks

[dim]Press [bold]Escape[/], [bold]Enter[/], or [bold]Q[/] to close[/]
""", id="shortcuts-content"),
            id="shortcuts-container"
        )
    
    def action_dismiss(self) -> None:
        self.app.pop_screen()


class ChatSession:
    """Represents a chat session"""
    
    def __init__(self, name: str, model: str = "llama3.2:1b"):
        self.name = name
        self.model = model
        self.messages = []
        self.created = datetime.now().isoformat()
        self.file_path = CHATS_DIR / f"{name.replace(' ', '_').replace(':', '-')}.json"
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save()
    
    def save(self):
        data = {
            "name": self.name,
            "model": self.model,
            "created": self.created,
            "messages": self.messages
        }
        self.file_path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, file_path: Path):
        data = json.loads(file_path.read_text())
        session = cls(data["name"], data.get("model", "llama3.2:1b"))
        session.messages = data.get("messages", [])
        session.created = data.get("created", datetime.now().isoformat())
        session.file_path = file_path
        return session


class AnimatedLoading(Static):
    """Animated loading indicator"""
    
    is_loading = reactive(False)
    frame_index = reactive(0)
    
    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._timer = None
    
    def on_mount(self) -> None:
        self._timer = self.set_interval(0.15, self.advance_frame)
    
    def advance_frame(self) -> None:
        if self.is_loading:
            self.frame_index = (self.frame_index + 1) % len(THINKING_ANIMATIONS)
            self.update(f"[bold magenta]{THINKING_ANIMATIONS[self.frame_index]}[/]")
        else:
            self.update("")
    
    def start(self) -> None:
        self.is_loading = True
    
    def stop(self) -> None:
        self.is_loading = False
        self.update("")


class ModelSelector(Container):
    """Collapsible model selector"""
    
    MODELS = [
        ("⚡ qwen2.5:0.5b", "qwen2.5:0.5b", "Fastest - 400 tok/sec"),
        ("⚡ tinyllama", "tinyllama", "Fast - 250 tok/sec"),
        ("⚡ llama3.2:1b", "llama3.2:1b", "Fast - Pre-installed"),
        ("⚡ llama3.2:3b", "llama3.2:3b", "Fast chat"),
        ("🎯 phi3:14b", "phi3:14b", "Fast reasoning"),
        ("🎯 gemma2:27b", "gemma2:27b", "General purpose"),
        ("💻 qwen2.5-coder:32b", "qwen2.5-coder:32b", "Best coding"),
        ("💻 deepseek-coder:33b", "deepseek-coder:33b", "Elite coding"),
        ("🚀 qwen2.5:72b", "qwen2.5:72b", "72B - CPU mode"),
        ("🚀 llama3.3:70b", "llama3.3:70b", "70B - CPU mode"),
    ]
    
    def compose(self) -> ComposeResult:
        with Collapsible(title="🤖 Models Ctrl+M", collapsed=False, id="models-collapsible"):
            yield OptionList(
                *[Option(f"{icon}  [dim]{desc}[/]", id=model) 
                  for icon, model, desc in self.MODELS],
                id="model-list"
            )


class SessionList(Container):
    """Collapsible session list"""
    
    def compose(self) -> ComposeResult:
        with Collapsible(title="📂 Sessions Ctrl+S", collapsed=False, id="sessions-collapsible"):
            yield Button("➕ New Chat", id="new-chat-btn", variant="primary")
            yield OptionList(id="session-list")


class ChatPanel(Container):
    """Main chat panel"""
    
    def compose(self) -> ComposeResult:
        yield Static("[bold]💬 Chat[/] [dim](? for shortcuts)[/]", id="chat-title")
        yield VerticalScroll(id="messages")
        yield AnimatedLoading(id="loading")
        yield Horizontal(
            Input(placeholder="Type message... (Enter to send)", id="chat-input"),
            Button("📤", id="send-btn", variant="success"),
            id="input-row"
        )


class LocalAI(App):
    """Local AI - tmux-style Terminal Chat Interface"""
    
    CSS = """
    Screen {
        layout: horizontal;
    }
    
    /* Modal shortcuts screen */
    ShortcutsScreen {
        align: center middle;
    }
    
    #shortcuts-container {
        width: 70;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    
    #shortcuts-content {
        width: 100%;
    }
    
    /* Sidebar - Sessions */
    SessionList {
        width: 28;
        background: $surface;
        border-right: tall $primary;
        padding: 1;
    }
    
    #new-chat-btn {
        width: 100%;
        margin: 1 0;
    }
    
    #session-list {
        height: 1fr;
        min-height: 5;
    }
    
    /* Model Selector */
    ModelSelector {
        width: 32;
        background: $surface;
        border-right: tall $secondary;
        padding: 1;
    }
    
    #model-list {
        height: 1fr;
        min-height: 5;
    }
    
    /* Collapsible styling */
    Collapsible {
        background: $surface;
        padding: 0;
    }
    
    CollapsibleTitle {
        background: $primary;
        padding: 1;
    }
    
    CollapsibleTitle:hover {
        background: $primary-lighten-1;
    }
    
    /* Chat Panel */
    ChatPanel {
        width: 1fr;
        padding: 1;
    }
    
    #chat-title {
        dock: top;
        padding: 1;
        background: $primary;
        text-align: center;
    }
    
    #messages {
        height: 1fr;
        padding: 1;
        background: $surface;
        border: round $primary;
    }
    
    #loading {
        height: 3;
        content-align: center middle;
        background: $surface;
    }
    
    #input-row {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }
    
    #chat-input {
        width: 1fr;
    }
    
    #send-btn {
        width: 6;
    }
    
    /* Messages */
    .user-msg {
        background: $primary 30%;
        padding: 1;
        margin: 1 0;
        border-left: thick $primary;
    }
    
    .ai-msg {
        background: $success 20%;
        padding: 1;
        margin: 1 0;
        border-left: thick $success;
    }
    
    .system-msg {
        color: $text-muted;
        text-align: center;
        padding: 1;
    }
    
    /* Option list styling */
    OptionList > .option-list--option {
        padding: 1;
    }
    
    OptionList > .option-list--option-highlighted {
        background: $secondary;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+n", "new_chat", "New Chat", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+s", "toggle_sessions", "Sessions"),
        Binding("ctrl+m", "toggle_models", "Models"),
        Binding("ctrl+1", "focus_sessions", "→Sessions"),
        Binding("ctrl+2", "focus_models", "→Models"),
        Binding("ctrl+3", "focus_chat", "→Chat"),
        Binding("tab", "cycle_focus", "Cycle"),
        Binding("escape", "focus_input", "Input"),
        Binding("question_mark", "show_shortcuts", "Help"),
        Binding("f1", "show_shortcuts", "Help"),
    ]
    
    current_session: reactive[ChatSession | None] = reactive(None)
    current_model: reactive[str] = reactive("llama3.2:1b")
    show_welcome: reactive[bool] = reactive(True)
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            SessionList(),
            ModelSelector(),
            ChatPanel(),
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize app and show shortcuts"""
        self.title = "Local AI"
        self.sub_title = "Press ? for shortcuts"
        self.load_sessions()
        
        # Show shortcuts on first open
        self.push_screen(ShortcutsScreen())
        
        self.query_one("#chat-input", Input).focus()
    
    def action_show_shortcuts(self) -> None:
        """Show shortcuts modal"""
        self.push_screen(ShortcutsScreen())
    
    def action_toggle_sessions(self) -> None:
        """Toggle sessions panel collapse"""
        collapsible = self.query_one("#sessions-collapsible", Collapsible)
        collapsible.collapsed = not collapsible.collapsed
    
    def action_toggle_models(self) -> None:
        """Toggle models panel collapse"""
        collapsible = self.query_one("#models-collapsible", Collapsible)
        collapsible.collapsed = not collapsible.collapsed
    
    def load_sessions(self) -> None:
        """Load chat sessions from disk"""
        session_list = self.query_one("#session-list", OptionList)
        session_list.clear_options()
        
        sessions = []
        for chat_file in sorted(CHATS_DIR.glob("*.json"), reverse=True):
            try:
                session = ChatSession.load(chat_file)
                sessions.append(session)
            except Exception:
                pass
        
        for session in sessions[:20]:
            short_name = session.name[:18] + "..." if len(session.name) > 20 else session.name
            session_list.add_option(Option(f"💬 {short_name}", id=str(session.file_path)))
    
    @on(Button.Pressed, "#new-chat-btn")
    def action_new_chat(self) -> None:
        """Create new chat session"""
        timestamp = datetime.now().strftime("%m-%d %H:%M")
        name = f"{self.current_model} {timestamp}"
        
        session = ChatSession(name, self.current_model)
        session.save()
        
        self.current_session = session
        self.load_sessions()
        self.update_chat_display()
        
        title = self.query_one("#chat-title", Static)
        title.update(f"[bold]💬 {name}[/]")
        
        self.notify(f"Created: {name}")
        self.query_one("#chat-input", Input).focus()
    
    @on(OptionList.OptionSelected, "#session-list")
    def on_session_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle session selection"""
        file_path = Path(event.option_id)
        if file_path.exists():
            self.current_session = ChatSession.load(file_path)
            self.current_model = self.current_session.model
            self.update_chat_display()
            
            title = self.query_one("#chat-title", Static)
            title.update(f"[bold]💬 {self.current_session.name}[/]")
            
            model_list = self.query_one("#model-list", OptionList)
            for i, (_, model, _) in enumerate(ModelSelector.MODELS):
                if model == self.current_model:
                    model_list.highlighted = i
                    break
            
            self.query_one("#chat-input", Input).focus()
    
    @on(OptionList.OptionSelected, "#model-list")
    def on_model_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle model selection"""
        self.current_model = event.option_id
        
        if self.current_session:
            self.current_session.model = self.current_model
            self.current_session.save()
        
        self.notify(f"Model: {self.current_model}")
        self.query_one("#chat-input", Input).focus()
    
    def update_chat_display(self) -> None:
        """Update chat messages"""
        messages = self.query_one("#messages", VerticalScroll)
        messages.remove_children()
        
        if self.current_session:
            for msg in self.current_session.messages:
                ts = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M")
                if msg["role"] == "user":
                    messages.mount(Static(
                        f"[bold cyan]You[/] [dim]{ts}[/]\n{msg['content']}",
                        classes="user-msg"
                    ))
                else:
                    messages.mount(Static(
                        f"[bold green]AI[/] [dim]{ts}[/]\n{msg['content']}",
                        classes="ai-msg"
                    ))
            messages.scroll_end()
    
    @on(Button.Pressed, "#send-btn")
    @on(Input.Submitted, "#chat-input")
    async def send_message(self) -> None:
        """Send message"""
        input_widget = self.query_one("#chat-input", Input)
        message = input_widget.value.strip()
        
        if not message:
            return
        
        if not self.current_session:
            self.action_new_chat()
        
        input_widget.value = ""
        
        self.current_session.add_message("user", message)
        self.update_chat_display()
        
        self.get_ai_response(message)
    
    @work(exclusive=True)
    async def get_ai_response(self, message: str) -> None:
        """Get AI response with loading animation"""
        loading = self.query_one("#loading", AnimatedLoading)
        
        loading.start()
        
        try:
            history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.current_session.messages
            ]
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": self.current_session.model,
                        "messages": history,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    ai_content = data["message"]["content"]
                    
                    self.current_session.add_message("assistant", ai_content)
                    self.call_from_thread(self.update_chat_display)
                else:
                    self.notify(f"Error: {response.status_code}", severity="error")
                    
        except httpx.ConnectError:
            self.notify("Cannot connect to Ollama!", severity="error")
        except Exception as e:
            self.notify(f"Error: {str(e)}", severity="error")
        finally:
            loading.stop()
    
    def action_focus_sessions(self) -> None:
        self.query_one("#session-list", OptionList).focus()
    
    def action_focus_models(self) -> None:
        self.query_one("#model-list", OptionList).focus()
    
    def action_focus_chat(self) -> None:
        self.query_one("#chat-input", Input).focus()
    
    def action_focus_input(self) -> None:
        self.query_one("#chat-input", Input).focus()
    
    def action_cycle_focus(self) -> None:
        """Cycle through panels"""
        widgets = [
            self.query_one("#session-list", OptionList),
            self.query_one("#model-list", OptionList),
            self.query_one("#chat-input", Input),
        ]
        
        current = self.focused
        for i, widget in enumerate(widgets):
            if widget == current or widget.has_focus:
                next_widget = widgets[(i + 1) % len(widgets)]
                next_widget.focus()
                return
        
        widgets[0].focus()


def main():
    app = LocalAI()
    app.run()


if __name__ == "__main__":
    main()
