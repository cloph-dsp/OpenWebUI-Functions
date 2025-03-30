"""
title: Pollinations OpenAI Pipe
author: kastru (based on Raiyan Hasan's work)
version: 1.0
license: MIT

Integrates Pollinations with OpenAI models for text and reasoning tasks.
Note: Temporary issue with Pollination's o3-mini reasoning model.
"""

import os
import json
import urllib.parse
from pydantic import BaseModel, Field
import requests
from typing import List, Union, Iterator, Dict, Any
import time

# Try to import OpenAI client for better handling of openai-reasoning model
try:
    from openai import OpenAI
    HAS_OPENAI_CLIENT = True
except ImportError:
    HAS_OPENAI_CLIENT = False
    if DEBUG:
        print("Warning: OpenAI client not available. Install with 'pip install openai'")

DEBUG = True
POLLINATIONS_BASE_API = "https://text.pollinations.ai"

class Pipe:
    def __init__(self):
        self.id = "pollinations-openai"
        self.type = "manifold"
        self.name = "OpenAI "
        self.api_key = "dummy-key"
        
        # Model configuration - use OpenAI endpoint for all models
        self.models = {
            "openai-large": {
                "name": "GPT-4o",
                "params": {"_reasoning": False, "_vision": True},
                "api_name": "openai-large",
                "endpoint_type": "openai"
            },
            "openai": {
                "name": "GPT-4o-mini",
                "params": {"_reasoning": False, "_vision": True},
                "api_name": "openai",
                "endpoint_type": "openai"
            },
            "openai-reasoning": {
                "name": "o3-mini",
                "params": {"_reasoning": True, "_vision": False},
                "api_name": "openai-reasoning",
                "endpoint_type": "openai"  # Changed back to openai endpoint
            },
        }

    def _get_model_config(self, model_id: str) -> dict:
        """Validate and return model config with force-flags"""
        # Ensure we're using the exact model ID that was requested
        if model_id not in self.models:
            if DEBUG:
                print(f"Warning: Unknown model ID '{model_id}'. Using openai-large.")
            model_id = "openai-large"
            
        config = self.models[model_id]
        
        if DEBUG:
            print(f"Using model config for '{model_id}': {json.dumps(config, indent=2)}")
            
        return {
            "api_name": config["api_name"],
            "params": {
                "reasoning": config["params"]["_reasoning"],
                "vision": config["params"]["_vision"],
            },
            "endpoint_type": config["endpoint_type"]
        }

    def _process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Process messages with model-specific handling"""
        processed = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = [
                    (
                        f"Image: {item['image_url']['url']}"
                        if item.get("type") == "image_url"
                        else item.get("text", "")
                    )
                    for item in content
                ]
                content = " ".join(parts)
            processed.append({"role": msg["role"], "content": content})
        return processed

    def _parse_sse_event(self, line: str) -> tuple:
        """Parse SSE event to extract content and model information"""
        if not line or line == "data: [DONE]":
            return "", ""
            
        if line.startswith("data: "):
            try:
                # Extract the JSON part after "data: "
                json_str = line[6:]
                data = json.loads(json_str)
                
                # Get model information
                model = data.get("model", "")
                
                # Get content from delta if available
                content = ""
                if "choices" in data and len(data["choices"]) > 0:
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                
                return content, model
            except json.JSONDecodeError:
                if DEBUG:
                    print(f"Failed to parse SSE event: {line}")
        
        return "", ""

    def pipe(self, body: Dict[str, Any]) -> Union[str, Iterator[str]]:
        try:
            # Explicitly get the model ID from the request and strip any prefixes like "openai1."
            model_id = body.get("model", "openai-large")
            
            # Clean model_id by removing any prefix before the dot (if present)
            if "." in model_id:
                model_id = model_id.split(".", 1)[1]
            
            if DEBUG:
                print(f"Original model_id from request: {model_id}")
            
            # Get model configuration
            config = self._get_model_config(model_id)
            
            # For openai-reasoning, try using the OpenAI client approach if available
            if model_id == "openai-reasoning" and HAS_OPENAI_CLIENT:
                return self._handle_openai_client(model_id, config, body)
            else:
                # Use standard approach for other models
                return self._handle_openai_endpoint(model_id, config, body)
                
        except Exception as e:
            if DEBUG:
                print(f"Pipeline Error: {str(e)}")
            return f"Error: {str(e)}"

    def _handle_openai_client(self, model_id, config, body):
        """Handle request using the OpenAI client (for reasoning model)"""
        try:
            if DEBUG:
                print("Using OpenAI client for openai-reasoning model")
            
            # Initialize OpenAI client
            client = OpenAI(
                api_key=self.api_key or "dummy-key",
                base_url=f"{POLLINATIONS_BASE_API}/openai",
            )
            
            # Process messages to match expected format
            messages = self._process_messages(body.get("messages", []))
            
            # Build kwargs for API call - only include parameters OpenAI client accepts
            kwargs = {
                "model": config["api_name"],
                "messages": messages,
                "temperature": body.get("temperature", 0.7),
                "max_tokens": body.get("max_tokens", 4096),
                "stream": body.get("stream", True)
            }
            
            # We can't pass custom parameters like 'reasoning' to the OpenAI client
            # Instead, we'll fall back to using the direct API approach
            if DEBUG:
                print("OpenAI client doesn't support custom parameters, using direct API approach")
            
            return self._handle_openai_endpoint(model_id, config, body)
            
        except Exception as e:
            if DEBUG:
                print(f"Error in _handle_openai_client: {e}")
            # Fall back to regular endpoint
            return self._handle_openai_endpoint(model_id, config, body)

    def _handle_openai_endpoint(self, model_id, config, body):
        """Handle request using the OpenAI-compatible endpoint"""
        # Following the exact example from pollination-example.py
        payload = {
            "model": config["api_name"],
            "messages": self._process_messages(body.get("messages", [])),
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 4096),
            "seed": body.get("seed", 42),
            "jsonMode": body.get("json_mode", False),
            "private": body.get("private", True),
            # Important: Use the stream setting from the request
            "stream": body.get("stream", True),
        }
        
        # Set model-specific parameters
        if model_id == "openai-reasoning":
            payload["reasoning"] = True
            # The example uses a boolean, but the documentation mentions reasoning_effort
            # Let's try both approaches
            payload["reasoning_effort"] = "high"
        else:
            if config["params"]["vision"]:
                payload["vision"] = True
            payload["reasoning"] = False
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }
        
        if DEBUG:
            print(f"Direct API request payload: {json.dumps({k: v for k, v in payload.items() if k != 'messages'})}")
        
        # Using the exact approach from pollination-example.py
        try:
            response = requests.post(
                f"{POLLINATIONS_BASE_API}/openai",
                json=payload,
                headers=headers,
                stream=body.get("stream", True),
                timeout=90,  # Longer timeout for reasoning model
            )
            
            # Check for error response
            if response.status_code != 200:
                error_msg = f"Error {response.status_code}: {response.text}"
                if DEBUG:
                    print(error_msg)
                return error_msg
            
            # Process response based on streaming preference
            if body.get("stream"):
                # Use the SSE parsing approach for all models
                return self._handle_streaming_sse(response, config, payload, model_id)
            else:
                # Following exactly how pollination-example.py handles non-streaming
                if payload.get("jsonMode"):
                    return response.json()
                else:
                    # Return the response directly without model verification details
                    return response.text
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if DEBUG:
                print(error_msg)
            return error_msg

    def _handle_streaming_sse(self, response, config, payload, original_model_id):
        """Parse SSE streaming response and extract only the content"""
        def generate():
            model_info = None
            buffer = ""
            
            # No model verification details - start yielding content immediately
            
            # Process the response stream
            for chunk in response.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                    
                text = chunk.decode('utf-8')
                buffer += text
                
                # Process buffer line by line (SSE format: data: {...}\n\n)
                lines = buffer.split("\n\n")
                buffer = lines.pop() if lines else ""  # Keep incomplete last line
                
                for line in lines:
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        
                        if data_str == "[DONE]":
                            if DEBUG:
                                print("Received [DONE] marker")
                            continue
                            
                        try:
                            data = json.loads(data_str)
                            
                            # Capture model info for debugging only
                            if not model_info and "model" in data:
                                model_info = data["model"]
                                if DEBUG:
                                    print(f"Detected model from response: {model_info}")
                            
                            # Extract content from delta
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            if DEBUG:
                                print(f"Failed to parse SSE event: {line}")
        
        return generate()

    def health_check(self) -> bool:
        """Test reasoning capability"""
        try:
            response = self.pipe(
                {
                    "model": "openai-reasoning",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Solve: A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does each cost?",
                        }
                    ],
                    "temperature": 0.0,
                }
            )
            return "ball: $0.05" in str(response).lower()
        except Exception as e:
            if DEBUG:
                print(f"Health Check Failed: {e}")
            return False

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": k, "name": v["name"]} for k, v in self.models.items()]