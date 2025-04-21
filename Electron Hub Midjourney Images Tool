"""
title: Electron Hub Midjourney Images
author: cloph-dsp
Github: https://github.com/cloph-dsp/OpenWebUI-Functions-and-Tools
date: 2025-04-25
version: 1.2.0
license: MIT
description: Generate images using Midjourney and Niji models via the Electron Hub API.
requirements: requests, pydantic
"""

import os
import asyncio
import requests
import re
import logging
from pydantic import BaseModel, Field
from typing import Literal, Awaitable, Callable, Optional, Dict, Any

# --- Setup Logging ---
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  # Set to DEBUG for more verbose output if needed
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not log.handlers:
    log.addHandler(handler)

# --- Constants ---
SIZE_MAP = {
    "1:1 (1024x1024)": "1024x1024",
    "1:2 (768x1536)": "768x1536",
    "2:1 (1536x768)": "1536x768",
    "2:3 (896x1344)": "896x1344",
    "3:2 (1344x896)": "1344x896",
    "3:4 (928x1232)": "928x1232",
    "4:3 (1232x928)": "1232x928",
    "9:16 (816x1456)": "816x1456",
    "16:9 (1456x816)": "1456x816",
}
ELECTRONHUB_IMAGE_API_URL = "https://api.electronhub.top/v1/images/generations"
DEFAULT_OPTIMIZER_SYSTEM_PROMPT = """
You are an expert prompt engineer for Midjourney and Niji models. Rewrite the user's idea into a detailed, production-ready prompt optimized for Midjourney v6/v7 or Niji. Focus on subject, scene, mood, style, lighting, composition, and colors. Add relevant parameters like `--ar` or `--style` if appropriate. Keep it concise (under 100 words). Output *only* the rewritten prompt.

Example transformation:
User: "A cat astronaut floating in space"
Optimized: "A fluffy orange cat in a detailed astronaut suit floating against a star-filled nebula backdrop::2, illuminated by soft blue rim light::1, photorealistic digital art --ar 16:9 --s 250 --c 15 --style raw"
"""


class Tools:
    # --- Valves ---
    class Valves(BaseModel):
        # Electron Hub Config
        electronhub_api_key: str = Field(
            default_factory=lambda: os.getenv("ELECTRONHUB_API_KEY", ""),
            description="Required: Your Electron Hub API key (ek-...).",
        )

        # Optional Prompt Optimization Config
        enable_prompt_optimization: bool = Field(
            default=False,
            description="Optional: Enhance prompts using a secondary LLM.",
        )
        optimizer_llm_base_url: str = Field(
            default="",
            description="Optional: Base URL for prompt optimizer LLM API.",
        )
        optimizer_llm_api_key: str = Field(
            default_factory=lambda: os.getenv("OPTIMIZER_LLM_API_KEY", ""),
            description="Optional: API Key for prompt optimizer LLM.",
        )
        optimizer_llm_model: str = Field(
            default="",
            description="Optional: Model name for prompt optimization.",
        )
        optimizer_system_prompt: str = Field(
            default=DEFAULT_OPTIMIZER_SYSTEM_PROMPT,
            description="Optional: System prompt for the optimizer LLM.",
            extra={"type": "textarea"},
        )

    def __init__(self):
        self.valves = self.Valves()

    # --- Helper: Optimize Prompt ---
    async def _optimize_prompt(
        self,
        original_prompt: str,
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> str:
        if not self.valves.enable_prompt_optimization:
            return original_prompt
        if not all(
            [
                self.valves.optimizer_llm_base_url,
                self.valves.optimizer_llm_api_key,
                self.valves.optimizer_llm_model,
            ]
        ):
            log.warning(
                "Prompt optimization enabled but optimizer LLM is not fully configured. Skipping."
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Skipping optimization: Optimizer not configured.",
                            "done": True,
                            "hidden": True,
                        },
                    }
                )
            return original_prompt

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Optimizing prompt...", "done": False},
                }
            )
        optimizer_url = (
            f"{self.valves.optimizer_llm_base_url.rstrip('/')}/chat/completions"
        )
        headers = {
            "Authorization": f"Bearer {self.valves.optimizer_llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.valves.optimizer_llm_model,
            "messages": [
                {"role": "system", "content": self.valves.optimizer_system_prompt},
                {"role": "user", "content": original_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        }
        try:
            response = await asyncio.to_thread(
                requests.post, optimizer_url, json=payload, headers=headers, timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            optimized_prompt = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if optimized_prompt:
                log.info(f"Raw optimized prompt: '{optimized_prompt}'")
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Prompt optimization complete.",
                                "done": True,
                                "hidden": True,
                            },
                        }
                    )
                return optimized_prompt
            else:
                raise ValueError("Optimizer LLM returned empty content.")
        except Exception as e:
            error_msg = f"Prompt optimization failed: {e}. Using original prompt."
            log.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return original_prompt

    # --- Helper: Sanitize Prompt for API ---
    def _sanitize_prompt(self, prompt_text: str) -> str:
        """Removes Midjourney parameters (like --ar) and cleans whitespace."""
        # Remove --parameter patterns
        pattern = r"--[a-zA-Z]+(?:\s+\S+)?"
        sanitized = re.sub(pattern, "", prompt_text)
        # Clean up whitespace
        sanitized = " ".join(sanitized.split()).strip()
        # Truncate if excessively long
        max_len = 450
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len] + "..."
            log.warning(f"Sanitized prompt was truncated to {max_len} characters.")
        return sanitized

    # --- Main Tool Function ---
    async def generate_midjourney_image(
        self,
        prompt: str,
        model: Literal[
            "midjourney-v7",
            "midjourney-v6.1",
            "midjourney-v6",
            "midjourney-v5.2",
            "midjourney-v5.1",
            "midjourney-v5",
            "niji-v6",
            "niji-v5",
        ] = "midjourney-v6.1",
        size: Literal[
            "1:1 (1024x1024)",
            "1:2 (768x1536)",
            "2:1 (1536x768)",
            "2:3 (896x1344)",
            "3:2 (1344x896)",
            "3:4 (928x1232)",
            "4:3 (1232x928)",
            "9:16 (816x1456)",
            "16:9 (1456x816)",
        ] = "1:1 (1024x1024)",
        n: Literal[1, 2, 3, 4] = 1,  # Allowed n=3
        quality: Literal["standard", "clarity", "hd", "ultra_hd"] = "standard",
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> str:
        """
        Generates images via Electron Hub using Midjourney/Niji models. Optionally optimizes and sanitizes the prompt first.

        Args:
            prompt (str): User's description of the desired image(s).
            model (Literal): Midjourney or Niji model version.
            size (Literal): Aspect ratio and approximate resolution.
            n (Literal): Number of images to generate (1, 2, 3, or 4).
            quality (Literal): Image quality setting ('midjourney-v7' only supports 'standard').
            __event_emitter__ (Callable): Optional status event emitter.

        Returns:
            str: Markdown for generated images or an error message.
        """
        # 1. Check Key
        if not self.valves.electronhub_api_key:
            error_msg = "Error: Electron Hub API key is missing. Configure via Valves or ELECTRONHUB_API_KEY env var."
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return f"Tool Error: {error_msg}"

        # 2. Optimize (Optional)
        potentially_optimized_prompt = await self._optimize_prompt(
            prompt, __event_emitter__
        )

        # 3. Sanitize
        final_api_prompt = self._sanitize_prompt(potentially_optimized_prompt)
        log.info(f"Sanitized prompt for API: '{final_api_prompt}'")

        # 4. Prepare Payload
        effective_quality = "standard" if model == "midjourney-v7" else quality
        headers = {
            "Authorization": f"Bearer {self.valves.electronhub_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "prompt": final_api_prompt,
            "n": n,
            "size": SIZE_MAP.get(size, "1024x1024"),
            "quality": effective_quality,
        }
        log.debug(f"Payload for Electron Hub: {payload}")

        # 5. Emit Status
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Generating image(s) with {model}...",
                        "done": False,
                    },
                }
            )

        # 6. Call API
        try:
            response = await asyncio.to_thread(
                requests.post,
                ELECTRONHUB_IMAGE_API_URL,
                json=payload,
                headers=headers,
                timeout=90,
            )
            response.raise_for_status()
            response_data = response.json()
            image_urls = [
                item.get("url")
                for item in response_data.get("data", [])
                if item.get("url")
            ]

            if not image_urls:
                error_msg = "API call successful, but no image URLs were returned."
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": error_msg, "done": True},
                        }
                    )
                return f"Tool Info: {error_msg}"

            # 7. Emit Status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Image generation complete!",
                            "done": True,
                            "hidden": True,
                        },
                    }
                )

            # 8. Return Result
            markdown_images = "\n".join(f"![Image]({url})" for url in image_urls)
            optimization_note = (
                " after enhancing prompt"
                if self.valves.enable_prompt_optimization
                else ""
            )
            return (
                f"Successfully generated {len(image_urls)} image(s) using {model}{optimization_note}.\n"
                f"Display the images using markdown:\n{markdown_images}"
            )

        except requests.exceptions.HTTPError as e:
            error_msg = f"API Error: {e.response.status_code} {e.response.reason}. Details: {e.response.text}"
            log.error(error_msg)
        except requests.exceptions.Timeout:
            error_msg = "Error: API request to Electron Hub timed out."
            log.error(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error calling Electron Hub: {e}"
            log.error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during generation: {e}"
            log.exception("Unexpected error in generate_midjourney_image")

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": error_msg.split(".")[0], "done": True},
                }
            )
        return f"Tool Error: {error_msg}"
