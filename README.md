# GeminiGTK

Native GTK+ 3 desktop client for exploring Gemini and Nano Banana models.

## Features

- Conversation sidebar with create/delete controls and persistent history stored on disk (`~/.gemini_gtk/conversations.json`).
- Main chat view that formats role, timestamp, and message content for readability.
- Bottom input bar with model selector, text entry, and send button.
- Real-time calls to Gemini / Nano Banana models via the official `google-genai` SDK with graceful error feedback.
- Lightweight Markdown-style rendering for model responses (headings, bold/italic, lists, code fences, and horizontal rules), plus LaTeX math rendering to inline images when `matplotlib` is available.

## Running locally

1. Install the runtime dependencies (Ubuntu/Debian example):

   ```bash
   sudo apt-get update && sudo apt-get install -y python3-gi gir1.2-gtk-3.0 python3-gi-cairo
   ```

2. Install the Gemini SDK:

   ```bash
   pip install google-genai
   ```

3. Export your API key (either `GEMINI_API_KEY` or `GOOGLE_GENAI_API_KEY` is accepted):

   ```bash
   export GEMINI_API_KEY="your-key-here"
   ```

4. Launch the application:

   ```bash
   python3 src/main.py
   ```

The first run will create the `~/.gemini_gtk` directory for storing conversations.

## Notes

- Set `GEMINI_API_KEY` or `GOOGLE_GENAI_API_KEY` to authenticate requests. The UI will surface a readable error message if the key is missing or invalid.
- Model options are listed in `DEFAULT_MODELS` inside `src/main.py` and can be adjusted to any model ID supported by your account (e.g., `gemini-1.5-flash`).
- The message view supports basic Markdown-inspired formatting:
  - Headings via `#`, `##`, or `###` prefixes
  - Bullet lists using `- `
  - Bold/italic via `**text**` and `*text*`
  - Code fences using triple backticks
  - Horizontal rules using `---` or `***`
- Inline LaTeX enclosed in `$...$` or block math with `$$...$$` is rendered to images when `matplotlib` is installed; otherwise the raw text is shown.
