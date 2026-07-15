"""A minimal agent: an OpenAI tool-calling loop, no custom framework."""

import json
from typing import List

from openai import OpenAI

from sherlock_ohms.tools import Tool

MAX_TOOL_ROUNDS = 5


class Agent:
    def __init__(self, client: OpenAI, model: str, instructions: str, tools: List[Tool], temperature: float = 0.0):
        self.client = client
        self.model = model
        self.instructions = instructions
        self.tools = {t.name: t for t in tools}
        self.temperature = temperature

    def run(self, query: str) -> dict:
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": query},
        ]

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                tools=[t.dict() for t in self.tools.values()],
                tool_choice="auto",
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                return {"answer": message.content, "messages": messages}

            for call in message.tool_calls:
                tool = self.tools.get(call.function.name)
                args = json.loads(call.function.arguments)
                result = tool(**args) if tool else f"Unknown tool: {call.function.name}"
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": str(result)}
                )

        return {"answer": None, "messages": messages, "error": "max tool rounds exceeded"}
