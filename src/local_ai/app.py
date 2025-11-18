"""
Local AI - tmux-style Terminal Chat Interface for Ollama
Features: Split panes, mouse support, animated loading
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Static, Input, Button, 
    ListView, ListItem, Label, Select, LoadingIndicator,
    TabbedContent, TabPane, OptionList
)
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.reactive import reactive
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

# Loading animation frames (ASCII art)
LOADING_FRAMES = [
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
]

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
    """Animated loading indicator with custom animations"""
    
    is_loading = reactive(False)
    frame_index = reactive(0)
    
    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._timer = None
    
    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self.advance_frame)
    
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
    """Clickable model selector with mouse support"""
    
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
        yield Static("[bold cyan]Select Model (click to choose)[/]", id="model-title")
        yield OptionList(
            *[Option(f"{icon}  [dim]{desc}[/]", id=model) 
              for icon, model, desc in self.MODELS],
            id="model-list"
        )


class ChatPanel(Container):
    """Main chat panel with messages"""
    
    def compose(self) -> ComposeResult:
        yield Static("[bold]💬 Chat[/]", id="chat-title")
        yield VerticalScroll(id="messages")
        yield AnimatedLoading(id="loading")
        yield Horizontal(
            Input(placeholder="Type message... (Enter to send)", id="chat-input"),
            Button("📤", id="send-btn", variant="success"),
            id="input-row"
        )


class SessionList(Container):
    """Sidebar with chat sessions - clickable"""
    
    def compose(self) -> ComposeResult:
        yield Static("[bold green]📂 Sessions[/]", id="sessions-title")
        yield Button("➕ New Chat", id="new-chat-btn", variant="primary")
        yield OptionList(id="session-list")


class LocalAI(App):
    """Local AI - tmux-style Terminal Chat Interface"""
    
    CSS = """
    Screen {
        layout: horizontal;
    }
    
    /* Sidebar - Sessions */
    SessionList {
        width: 25;
        background: $surface;
        border-right: tall $primary;
        padding: 1;
    }
    
    #sessions-title {
        text-align: center;
        padding: 1;
        background: $success;
        color: $text;
        text-style: bold;
    }
    
    #new-chat-btn {
        width: 100%;
        margin: 1 0;
    }
    
    #session-list {
        height: 1fr;
        background: $surface;
    }
    
    #session-list > .option-list--option {
        padding: 1;
    }
    
    #session-list > .option-list--option-highlighted {
        background: $primary;
    }
    
    /* Model Selector */
    ModelSelector {
        width: 30;
        background: $surface;
        border-right: tall $secondary;
        padding: 1;
    }
    
    #model-title {
        text-align: center;
        padding: 1;
        background: $secondary;
        color: $text;
    }
    
    #model-list {
        height: 1fr;
    }
    
    #model-list > .option-list--option {
        padding: 1;
    }
    
    #model-list > .option-list--option-highlighted {
        background: $secondary;
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
    """
    
    BINDINGS = [
        Binding("ctrl+n", "new_chat", "New Chat", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+1", "focus_sessions", "Sessions"),
        Binding("ctrl+2", "focus_models", "Models"),
        Binding("ctrl+3", "focus_chat", "Chat"),
        Binding("tab", "cycle_focus", "Cycle Panels"),
        Binding("escape", "focus_input", "Input"),
    ]
    
    current_session: reactive[ChatSession | None] = reactive(None)
    current_model: reactive[str] = reactive("llama3.2:1b")
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            SessionList(),
            ModelSelector(),
            ChatPanel(),
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize app"""
        self.title = "Local AI"
        self.sub_title = "tmux-style Chat Interface"
        self.load_sessions()
        
        # Add welcome message
        messages = self.query_one("#messages", VerticalScroll)
        messages.mount(Static(
            "[dim]Welcome to Local AI! Click a model on the left, "
            "then start a new chat or select an existing one.[/]",
            classes="system-msg"
        ))
        
        self.query_one("#chat-input", Input).focus()
    
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
        
        for session in sessions[:20]:  # Limit to 20 recent
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
        
        # Update title
        title = self.query_one("#chat-title", Static)
        title.update(f"[bold]💬 {name}[/]")
        
        self.notify(f"Created: {name}")
        self.query_one("#chat-input", Input).focus()
    
    @on(OptionList.OptionSelected, "#session-list")
    def on_session_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle session selection via mouse click"""
        file_path = Path(event.option_id)
        if file_path.exists():
            self.current_session = ChatSession.load(file_path)
            self.current_model = self.current_session.model
            self.update_chat_display()
            
            # Update title
            title = self.query_one("#chat-title", Static)
            title.update(f"[bold]💬 {self.current_session.name}[/]")
            
            # Highlight model in list
            model_list = self.query_one("#model-list", OptionList)
            for i, (_, model, _) in enumerate(ModelSelector.MODELS):
                if model == self.current_model:
                    model_list.highlighted = i
                    break
            
            self.query_one("#chat-input", Input).focus()
    
    @on(OptionList.OptionSelected, "#model-list")
    def on_model_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle model selection via mouse click"""
        self.current_model = event.option_id
        
        if self.current_session:
            self.current_session.model = self.current_model
            self.current_session.save()
        
        self.notify(f"Model: {self.current_model}")
        self.query_one("#chat-input", Input).focus()
    
    def update_chat_display(self) -> None:
        """Update chat messages display"""
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
        """Send message to AI"""
        input_widget = self.query_one("#chat-input", Input)
        message = input_widget.value.strip()
        
        if not message:
            return
        
        if not self.current_session:
            self.action_new_chat()
        
        input_widget.value = ""
        
        # Add user message
        self.current_session.add_message("user", message)
        self.update_chat_display()
        
        # Get AI response with animation
        self.get_ai_response(message)
    
    @work(exclusive=True)
    async def get_ai_response(self, message: str) -> None:
        """Get response from Ollama with loading animation"""
        loading = self.query_one("#loading", AnimatedLoading)
        messages = self.query_one("#messages", VerticalScroll)
        
        # Start loading animation
        loading.start()
        
        try:
            # Build conversation history
            history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.current_session.messages
            ]
            
            # Call Ollama API
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
                    
                    # Save and display
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
        """Cycle through panels with Tab"""
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
