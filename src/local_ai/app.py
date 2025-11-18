"""
Local AI - Terminal Chat Interface for Ollama
A tmux-style TUI for chatting with local AI models
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Static, Input, Button, 
    ListView, ListItem, Label, Select, RichLog
)
from textual.binding import Binding
from textual.reactive import reactive
from textual import on
from datetime import datetime
from pathlib import Path
import json
import httpx
import asyncio

# Chat storage directory
CHATS_DIR = Path.home() / "Models" / "chats"
CHATS_DIR.mkdir(parents=True, exist_ok=True)


class ChatMessage(Static):
    """A single chat message"""
    
    def __init__(self, role: str, content: str, timestamp: str = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%H:%M")
        super().__init__()
    
    def compose(self) -> ComposeResult:
        if self.role == "user":
            yield Static(
                f"[bold cyan]You[/] [dim]{self.timestamp}[/]\n{self.content}",
                classes="user-message"
            )
        else:
            yield Static(
                f"[bold green]AI[/] [dim]{self.timestamp}[/]\n{self.content}",
                classes="ai-message"
            )


class ChatSession:
    """Represents a chat session"""
    
    def __init__(self, name: str, model: str = "llama3.2:1b"):
        self.name = name
        self.model = model
        self.messages = []
        self.created = datetime.now().isoformat()
        self.file_path = CHATS_DIR / f"{name.replace(' ', '_')}.json"
    
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


class Sidebar(Container):
    """Sidebar with chat sessions"""
    
    def compose(self) -> ComposeResult:
        yield Static("[bold]Chat Sessions[/]", id="sidebar-title")
        yield Button("+ New Chat", id="new-chat-btn", variant="primary")
        yield ListView(id="chat-list")


class ChatPanel(Container):
    """Main chat panel"""
    
    def compose(self) -> ComposeResult:
        yield Static("[bold]Select or create a chat[/]", id="chat-header")
        yield VerticalScroll(id="messages")
        yield Horizontal(
            Input(placeholder="Type your message...", id="chat-input"),
            Button("Send", id="send-btn", variant="success"),
            id="input-row"
        )


class ModelSelector(Container):
    """Model selection panel"""
    
    MODELS = [
        ("llama3.2:1b", "Llama 3.2 1B (Fast)"),
        ("llama3.2:3b", "Llama 3.2 3B"),
        ("qwen2.5:0.5b", "Qwen 2.5 0.5B (Fastest)"),
        ("tinyllama", "TinyLlama 1.1B"),
        ("phi3:14b", "Phi-3 14B"),
        ("gemma2:27b", "Gemma 2 27B"),
        ("qwen2.5-coder:32b", "Qwen 2.5 Coder 32B"),
        ("deepseek-coder:33b", "DeepSeek Coder 33B"),
        ("qwen2.5:72b", "Qwen 2.5 72B (CPU)"),
        ("llama3.3:70b", "Llama 3.3 70B (CPU)"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Static("[bold]Model:[/] ", id="model-label")
        yield Select(
            [(name, value) for value, name in self.MODELS],
            id="model-select",
            value="llama3.2:1b"
        )


class LocalAI(App):
    """Local AI - Terminal Chat Interface"""
    
    CSS = """
    Screen {
        layout: horizontal;
    }
    
    Sidebar {
        width: 30;
        background: $surface;
        border-right: solid $primary;
        padding: 1;
    }
    
    #sidebar-title {
        text-align: center;
        padding: 1;
        background: $primary;
        color: $text;
    }
    
    #new-chat-btn {
        width: 100%;
        margin: 1 0;
    }
    
    #chat-list {
        height: 1fr;
    }
    
    ChatPanel {
        width: 1fr;
        padding: 1;
    }
    
    #chat-header {
        dock: top;
        padding: 1;
        background: $surface;
        border-bottom: solid $primary;
    }
    
    #messages {
        height: 1fr;
        padding: 1;
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
        width: 10;
    }
    
    .user-message {
        background: $primary 20%;
        padding: 1;
        margin: 1 0;
        border: solid $primary;
    }
    
    .ai-message {
        background: $success 20%;
        padding: 1;
        margin: 1 0;
        border: solid $success;
    }
    
    ModelSelector {
        dock: top;
        height: 3;
        layout: horizontal;
        padding: 0 1;
        background: $surface;
    }
    
    #model-label {
        width: auto;
        padding: 1 0;
    }
    
    #model-select {
        width: 1fr;
    }
    
    ListItem {
        padding: 1;
    }
    
    ListItem:hover {
        background: $primary 30%;
    }
    
    ListItem.-selected {
        background: $primary;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+n", "new_chat", "New Chat"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+m", "toggle_model", "Switch Model"),
        Binding("ctrl+s", "save_chat", "Save"),
        Binding("escape", "focus_input", "Focus Input"),
    ]
    
    current_session: reactive[ChatSession | None] = reactive(None)
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ModelSelector()
        yield Horizontal(
            Sidebar(),
            ChatPanel(),
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Load existing chats on startup"""
        self.load_chat_list()
        self.query_one("#chat-input", Input).focus()
    
    def load_chat_list(self) -> None:
        """Load chat sessions from disk"""
        chat_list = self.query_one("#chat-list", ListView)
        chat_list.clear()
        
        for chat_file in sorted(CHATS_DIR.glob("*.json"), reverse=True):
            try:
                session = ChatSession.load(chat_file)
                item = ListItem(Label(session.name))
                item.data = session
                chat_list.append(item)
            except Exception:
                pass
    
    @on(Button.Pressed, "#new-chat-btn")
    def action_new_chat(self) -> None:
        """Create a new chat session"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        name = f"Chat {timestamp}"
        model = self.query_one("#model-select", Select).value
        
        session = ChatSession(name, model)
        session.save()
        
        self.current_session = session
        self.load_chat_list()
        self.update_chat_display()
        
        # Update header
        header = self.query_one("#chat-header", Static)
        header.update(f"[bold]{name}[/] - [dim]{model}[/]")
    
    @on(ListView.Selected, "#chat-list")
    def on_chat_selected(self, event: ListView.Selected) -> None:
        """Handle chat selection"""
        if hasattr(event.item, 'data'):
            self.current_session = event.item.data
            self.update_chat_display()
            
            # Update header
            header = self.query_one("#chat-header", Static)
            header.update(f"[bold]{self.current_session.name}[/] - [dim]{self.current_session.model}[/]")
            
            # Update model selector
            model_select = self.query_one("#model-select", Select)
            model_select.value = self.current_session.model
    
    def update_chat_display(self) -> None:
        """Update the chat messages display"""
        messages_container = self.query_one("#messages", VerticalScroll)
        messages_container.remove_children()
        
        if self.current_session:
            for msg in self.current_session.messages:
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M")
                messages_container.mount(
                    ChatMessage(msg["role"], msg["content"], timestamp)
                )
            messages_container.scroll_end()
    
    @on(Button.Pressed, "#send-btn")
    @on(Input.Submitted, "#chat-input")
    async def send_message(self) -> None:
        """Send a message to the AI"""
        input_widget = self.query_one("#chat-input", Input)
        message = input_widget.value.strip()
        
        if not message:
            return
        
        if not self.current_session:
            self.action_new_chat()
        
        # Clear input
        input_widget.value = ""
        
        # Add user message
        self.current_session.add_message("user", message)
        self.update_chat_display()
        
        # Get AI response
        await self.get_ai_response(message)
    
    async def get_ai_response(self, message: str) -> None:
        """Get response from Ollama"""
        messages_container = self.query_one("#messages", VerticalScroll)
        
        # Add placeholder for AI response
        ai_message = Static("[dim]Thinking...[/]", classes="ai-message")
        messages_container.mount(ai_message)
        messages_container.scroll_end()
        
        try:
            # Build conversation history
            history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.current_session.messages[:-1]  # Exclude the just-added message
            ]
            history.append({"role": "user", "content": message})
            
            # Call Ollama API
            async with httpx.AsyncClient(timeout=120.0) as client:
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
                    
                    # Update message
                    ai_message.update(
                        f"[bold green]AI[/] [dim]{datetime.now().strftime('%H:%M')}[/]\n{ai_content}"
                    )
                    
                    # Save to session
                    self.current_session.add_message("assistant", ai_content)
                else:
                    ai_message.update(f"[red]Error: {response.status_code}[/]")
                    
        except httpx.ConnectError:
            ai_message.update("[red]Error: Cannot connect to Ollama. Is it running?[/]")
        except Exception as e:
            ai_message.update(f"[red]Error: {str(e)}[/]")
        
        messages_container.scroll_end()
    
    @on(Select.Changed, "#model-select")
    def on_model_changed(self, event: Select.Changed) -> None:
        """Handle model selection change"""
        if self.current_session:
            self.current_session.model = event.value
            self.current_session.save()
            
            # Update header
            header = self.query_one("#chat-header", Static)
            header.update(f"[bold]{self.current_session.name}[/] - [dim]{self.current_session.model}[/]")
    
    def action_focus_input(self) -> None:
        """Focus the input field"""
        self.query_one("#chat-input", Input).focus()
    
    def action_toggle_model(self) -> None:
        """Toggle model selector"""
        self.query_one("#model-select", Select).focus()
    
    def action_save_chat(self) -> None:
        """Save current chat"""
        if self.current_session:
            self.current_session.save()
            self.notify("Chat saved!")


def main():
    app = LocalAI()
    app.run()


if __name__ == "__main__":
    main()
