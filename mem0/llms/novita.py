import os
from typing import Optional, List, Dict

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from openai import OpenAI

NOVITA_BASE_URL = "https://api.novita.ai/v3/openai"
MISTRA_NEMO_MODEL = "mistralai/mistral-nemo"


class NovitaLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = MISTRA_NEMO_MODEL

        if os.environ.get("NOVITA_API_KEY"):
            self.client = OpenAI(
                base_url=NOVITA_BASE_URL,
                api_key=os.environ.get("NOVITA_API_KEY"),
            )
        else:
            raise ValueError("NOVITA_API_KEY environment variable is not set.")


    def generate_response(self, messages: List[Dict[str, str]]):
        chat_completion_res = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=False,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        content = chat_completion_res.choices[0].message.content
        return content
