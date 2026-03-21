"""
Generic LLM client — unified interface for any chat-completion API.

Supports any OpenAI-compatible endpoint out of the box:
  - Ollama       (http://localhost:11434/v1)
  - OpenAI       (https://api.openai.com/v1)
  - Claude (via proxy or openai-compatible adapter)
  - vLLM, LM Studio, llama.cpp server, etc.

Usage:
    from limbiq.llm import LLMClient

    llm = LLMClient(base_url="http://localhost:11434/v1", model="llama3.1")
    response = llm("Summarize this text: ...")
    # or with messages:
    response = llm.chat([{"role": "user", "content": "Hello"}])
"""

import json
import logging
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Lightweight LLM client using only stdlib (no SDK dependencies).

    Works with any OpenAI-compatible chat/completions endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.1",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        """
        Simple call interface — takes a prompt string, returns a string.

        This is the signature limbiq expects for llm_fn:
            fn(prompt: str) -> str
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)

    def chat(self, messages: list[dict]) -> str:
        """
        Send a chat completion request and return the assistant's text.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            The assistant's response text.
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers=headers, method="POST")

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except URLError as e:
            logger.error(f"LLM request failed: {e}")
            raise ConnectionError(
                f"Could not connect to LLM at {url}. "
                f"Is the server running? Error: {e}"
            ) from e
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected LLM response format: {e}")
            raise ValueError(f"Unexpected response from LLM: {e}") from e

    def is_available(self) -> bool:
        """Check if the LLM endpoint is reachable."""
        try:
            url = f"{self.base_url}/models"
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = Request(url, headers=headers)
            with urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def __repr__(self):
        return f"LLMClient(base_url={self.base_url!r}, model={self.model!r})"
