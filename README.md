# Local AI

A beautiful terminal chat interface for local AI models using Ollama.

![Local AI Screenshot](docs/screenshot.png)

## Features

- **tmux-style Interface**: Sidebar with chat sessions, main chat panel
- **Multiple Models**: Switch between any Ollama model on the fly
- **Persistent Chats**: All conversations saved as JSON/Markdown
- **Keyboard Shortcuts**: vim-inspired navigation
- **Beautiful UI**: Catppuccin-themed with Rich text formatting
- **32GB VRAM Optimized**: Pre-configured for optimal model selection

## Installation

### From Source (Development)

```bash
git clone https://github.com/fivelidz/local-ai.git
cd local-ai
pip install -e .
```

### Requirements

- Python 3.10+
- Ollama running locally
- Terminal with true color support (Ghostty, Kitty, etc.)

## Usage

```bash
# Start Local AI
local-ai

# Or run directly
python -m local_ai.app
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+N` | New Chat |
| `Ctrl+M` | Switch Model |
| `Ctrl+S` | Save Chat |
| `Ctrl+Q` | Quit |
| `Escape` | Focus Input |
| `Enter` | Send Message |

## Available Models

### Recommended (32GB VRAM)
- `qwen2.5-coder:32b` - Best coding model
- `gemma2:27b` - General purpose
- `phi3:14b` - Fast reasoning

### Fast Writing
- `qwen2.5:0.5b` - 400 tokens/sec
- `tinyllama` - 250 tokens/sec
- `llama3.2:3b` - Fast chat

### Large Models (CPU)
- `qwen2.5:72b` - Multilingual, math
- `llama3.3:70b` - General flagship
- `deepseek-coder:33b` - Elite coding

## Chat Storage

Chats are saved to `~/Models/chats/` as JSON files:

```
~/Models/chats/
├── Chat_2025-11-18_10-30.json
├── Chat_2025-11-18_11-45.json
└── ...
```

Each file contains:
- Chat name
- Model used
- Full conversation history
- Timestamps

## Configuration

Configuration is stored in `~/.config/local-ai/config.json`:

```json
{
  "default_model": "llama3.2:1b",
  "theme": "catppuccin-mocha",
  "ollama_url": "http://localhost:11434",
  "chats_dir": "~/Models/chats"
}
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run in development mode
textual run --dev src/local_ai/app.py

# Run tests
pytest
```

## Architecture

```
local-ai/
├── src/local_ai/
│   ├── __init__.py
│   ├── app.py          # Main Textual application
│   ├── widgets.py      # Custom widgets
│   ├── models.py       # Chat session models
│   └── api.py          # Ollama API client
├── chats/              # Default chat storage
├── docs/
├── tests/
├── pyproject.toml
└── README.md
```

## Inspired By

- [oterm](https://github.com/ggozad/oterm) - Ollama terminal
- [parllama](https://github.com/paulrobello/parllama) - PAR LLAMA TUI
- [lit-tui](https://pypi.org/project/lit-tui/) - Lightweight TUI
- [lazygit](https://github.com/jesseduffield/lazygit) - tmux-style interface

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

Made with ❤️ for qalarc_OS
