import os
import json
import re
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# override=True makes the .env file win over any variables already set in your shell
load_dotenv(override=True)

REQUIRED = ["AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "AZURE_API_VERSION"]


class LLM:
    """Thin wrapper around Azure OpenAI (gpt-5-mini) using langchain-openai."""

    def __init__(self):
        missing = [k for k in REQUIRED if not os.getenv(k)]
        if missing:
            raise EnvironmentError(f"Missing in .env: {', '.join(missing)}")

        self.endpoint = os.getenv("AZURE_ENDPOINT").strip()
        self.deployment = os.getenv("AZURE_DEPLOYMENT").strip()
        self.api_version = os.getenv("AZURE_API_VERSION").strip()

        self._client = AzureChatOpenAI(
            api_key=os.getenv("AZURE_API_KEY").strip(),
            azure_endpoint=self.endpoint,
            azure_deployment=self.deployment,
            api_version=self.api_version,
            max_completion_tokens=8000,
            timeout=180,
            max_retries=2,
        )

    def describe(self) -> str:
        """Show the URL that will be called (no key), useful for debugging 404s."""
        return (
            f"{self.endpoint.rstrip('/')}/openai/deployments/"
            f"{self.deployment}/chat/completions?api-version={self.api_version}"
        )

    def complete(self, system: str, user: str) -> str:
        """Send a system + user prompt, return the text reply."""
        messages = [("system", system), ("human", user)]
        text = self._client.invoke(messages).content
        if not text or not text.strip():
            raise RuntimeError(
                "Model returned an empty reply (reasoning likely used the whole "
                "token budget). Raise max_completion_tokens in llm.py."
            )
        return text


def parse_json(text: str) -> dict:
    """Extract a JSON object from model output, even if wrapped in ```json fences."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


