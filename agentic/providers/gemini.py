import os
from typing import Optional, List, Any

from google import genai
from google.genai import types

from agentic import ModelProvider, Message, MessageRole


class GeminiModelProvider(ModelProvider):
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        # SDK  Initialization
        self.client = genai.Client(api_key=api_key)

    def generate(self, messages: List[Message], **kwargs: Any) -> Message:
        prompt = "\n".join([f"{m.role}: {m.content}" for m in messages])
        
        # Config Parameters can be passed via kwargs
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**kwargs) if kwargs else None
        )
        
        return Message(role=MessageRole.ASSISTANT, content=response.text)

    def stream(
            self,
            messages: List[Message],
            tools: Optional[List[Any]] = None,
            **kwargs: Any,
    ):
        """Stream responses from Gemini."""
        history_text = []
        for msg in messages:

            prefix = msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role)
            history_text.append(f"{prefix}: {msg.content}")
        prompt = "\n".join(history_text)


        generation_config = genai.types.GenerationConfig(**kwargs) if kwargs else None


        responses = self._model.generate_content(
            prompt,
            stream=True,
            generation_config=generation_config
        )

        for chunk in responses:
            text = getattr(chunk, "text", "") or ""
            if text:
                yield Message(role=MessageRole.ASSISTANT, content=text)