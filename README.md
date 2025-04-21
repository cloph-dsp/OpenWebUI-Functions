# OpenWebUI Functions & Tools

Import all of these directly from OpenWebUI for easy use in your workflows:  
https://openwebui.com/u/kastru

## Functions
- **Pollinations OpenAI Pipe**  
  File: `OpenAI Models Function.py`  
  Integrates Pollinations with OpenAI models (GPT‑4o, GPT‑4o‑mini, reasoning).  
  Supports streaming SSE, JSON/non‑stream responses, and model health checks.

- **OCR.Space Text Extraction**  
  File: `OCR-Space Function.py`  
  Extracts text from image URLs or base64 payloads via OCR.Space API.  
  Retries with exponential backoff and emits status events in `inlet`.

## Tools
- **Electron Hub Midjourney Images**  
  File: `Electron Hub Midjourney Images Tool.py`  
  Generates Midjourney & Niji images via Electron Hub API.  
  Optional prompt optimization, sanitization, and markdown output.
