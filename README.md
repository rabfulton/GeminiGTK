# GeminiGTK

Native GTK+ 3 desktop client for exploring Gemini and Nano Banana models.

## Features

- Conversation sidebar with create/delete controls and persistent history stored on disk (`~/.gemini_gtk/conversations.json`).
- Main chat view that formats role, timestamp, and message content for readability.
- Bottom input bar with model selector, text entry, and send button.
- Stubbed model replies that make it easy to wire in real Gemini / Nano Banana API calls later.

## Running locally

1. Install the runtime dependencies (Ubuntu/Debian example):

   ```bash
   sudo apt-get update && sudo apt-get install -y python3-gi gir1.2-gtk-3.0 python3-gi-cairo
   ```

2. Launch the application:

   ```bash
   python3 src/main.py
   ```

The first run will create the `~/.gemini_gtk` directory for storing conversations.

## Wiring up real model calls

`src/main.py` currently returns a placeholder response. Replace `_fake_model_response` with calls to your Gemini / Nano Banana SDK or REST API client, and store the returned text in `assistant_msg.content`.
