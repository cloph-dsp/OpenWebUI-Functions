"""
title: OCR.Space Text Extraction
author: cloph-dsp
version: 1.0
description: Extract text from images in chat messages using OCR.Space API. 
             Get your key here -> https://ocr.space/OCRAPI
"""

from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional
import asyncio
import base64
import json
import requests
import re


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for filtering operations."
        )
        API_KEY: str = Field(
            default="", description="OCR.Space API Key"
        )
        MAX_RETRIES: int = Field(default=3, description="Maximum retry attempts for OCR processing")
        LANGUAGE: str = Field(
            default="eng", 
            description="OCR language code (e.g., eng, ger, spa, fra, ita, por, etc.)"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _extract_base64_content(self, image_url: str) -> str:
        """Extract base64 content from data URL."""
        # Check if it's a data URL (base64)
        match = re.match(r'data:image/([a-zA-Z]+);base64,(.+)', image_url)
        if match:
            return match.group(2)
        return image_url

    async def _process_image_ocr(self, image_url: str, event_emitter) -> str:
        """Process image through OCR.Space API with retry mechanism."""
        retries = 0
        while retries < self.valves.MAX_RETRIES:
            try:
                # Extract base64 content if it's a data URL
                image_data = self._extract_base64_content(image_url)
                
                # Determine if it's a URL or base64 data
                if image_data == image_url:  # It's a URL
                    ocr_result = requests.post(
                        'https://api.ocr.space/parse/image',
                        data={
                            'url': image_url,
                            'apikey': self.valves.API_KEY,
                            'language': self.valves.LANGUAGE,
                            'isOverlayRequired': False
                        },
                    )
                else:  # It's base64 data
                    ocr_result = requests.post(
                        'https://api.ocr.space/parse/image',
                        data={
                            'base64Image': f'data:image/jpeg;base64,{image_data}',
                            'apikey': self.valves.API_KEY,
                            'language': self.valves.LANGUAGE,
                            'isOverlayRequired': False
                        },
                    )
                
                result = json.loads(ocr_result.content.decode())
                
                if result.get("OCRExitCode") == 1:  # Success
                    extracted_text = ""
                    for text_result in result.get("ParsedResults", []):
                        extracted_text += text_result.get("ParsedText", "")
                    
                    if not extracted_text:
                        raise Exception("No text extracted from the image")
                    
                    return extracted_text
                else:
                    error_message = result.get("ErrorMessage", "Unknown OCR error")
                    raise Exception(f"OCR error: {error_message}")
                
            except Exception as e:
                retries += 1
                if retries < self.valves.MAX_RETRIES:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"⚠️ OCR processing failed, retrying ({retries}/{self.valves.MAX_RETRIES})...",
                                "done": False,
                            },
                        }
                    )
                    await asyncio.sleep(2**retries)  # Exponential backoff
                else:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"❌ OCR failed after {self.valves.MAX_RETRIES} attempts: {str(e)}",
                                "done": True,
                            },
                        }
                    )
                    raise
        
        raise Exception(f"Failed to process image after {self.valves.MAX_RETRIES} attempts")

    def _find_image_in_messages(self, messages):
        """Find images in the user messages."""
        for m_index, message in enumerate(messages):
            if message["role"] == "user" and isinstance(message.get("content"), list):
                for c_index, content in enumerate(message["content"]):
                    if content["type"] == "image_url":
                        return m_index, c_index, content["image_url"]["url"]
        return None

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        messages = body.get("messages", [])
        
        # Check for API key
        if not self.valves.API_KEY or self.valves.API_KEY == "helloworld":
            # Add a system message informing about missing API key
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "❌ OCR feature disabled: No valid API key configured",
                        "done": True,
                    },
                }
            )
            # Add a message to the conversation
            if messages and messages[-1]["role"] == "user":
                content = messages[-1].get("content", "")
                if isinstance(content, list):
                    # Add a text note about the missing API key to the existing content
                    for item in content:
                        if item["type"] == "image_url":
                            messages[-1]["content"] = [
                                {"type": "text", "text": "I notice you've uploaded an image, but OCR processing is not available (missing API key)."}
                            ]
            return body
        
        # Look for images in messages
        image_info = self._find_image_in_messages(messages)
        if not image_info:
            return body
        
        message_index, content_index, image_url = image_info
        
        try:
            # Show processing status
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "🔍 Processing image with OCR.Space...",
                        "done": False,
                    },
                }
            )
            
            # Extract text from image
            extracted_text = await self._process_image_ocr(image_url, __event_emitter__)
            
            # Update status to complete
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "✅ Text successfully extracted from image",
                        "done": True,
                    },
                }
            )
            
            # Format the extracted text with context
            ocr_context = (
                f"[System note: The following text was extracted from an uploaded image via OCR. "
                f"Use it to inform your response.]\n\n{extracted_text}"
            )
            
            # Update the message content
            user_message = ""
            if isinstance(messages[message_index]["content"], list):
                for item in messages[message_index]["content"]:
                    if item["type"] == "text":
                        user_message = item["text"]
                        break
            
            # Replace the image with the extracted text
            messages[message_index]["content"] = [
                {
                    "type": "text",
                    "text": f"{ocr_context}\n\nUser query: {user_message}" if user_message else ocr_context
                }
            ]
            
            body["messages"] = messages
            
        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"❌ OCR processing failed: {str(e)}",
                        "done": True,
                    },
                }
            )
            
            # Replace the image with an error message
            if isinstance(messages[message_index]["content"], list):
                user_message = ""
                for item in messages[message_index]["content"]:
                    if item["type"] == "text":
                        user_message = item["text"]
                        break
                
                messages[message_index]["content"] = [
                    {
                        "type": "text",
                        "text": f"[Image processing failed: {str(e)}]\n\n{user_message}" if user_message else f"[Image processing failed: {str(e)}]"
                    }
                ]
                
                body["messages"] = messages
        
        return body

    async def stream(self, event: dict) -> dict:
        # No modifications needed for streaming events
        return event

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        # No modifications needed for the response
        return body
