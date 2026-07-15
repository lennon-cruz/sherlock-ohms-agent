"""Interactive REPL for the Sherlock Ohms agent.

Usage: uv run agent
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from sherlock_ohms import rag, vectorstore
from sherlock_ohms.agent import Agent
from sherlock_ohms.config import CORPORA, MODEL
from sherlock_ohms.tools import tool

INSTRUCTIONS = (
    "You are Sherlock Ohms, an assistant that can intelligently decide which tool to use "
    "to answer user questions. Route fintech questions to the fintech report search tool and "
    "telecom questions to the telecom report search tool. If a question doesn't match either "
    "report, say you don't have information on that topic instead of guessing."
)


def build_agent(client: OpenAI) -> Agent:
    api_key = os.environ["OPENAI_API_KEY"]
    fintech_collection = vectorstore.get_or_create_collection(CORPORA["fintech"]["collection_name"], api_key)
    telecom_collection = vectorstore.get_or_create_collection(CORPORA["telecom"]["collection_name"], api_key)

    @tool
    def search_fintech_report(query: str) -> str:
        """Search the fintech industry report for information about financial technology, digital
        payments, regulation, and market trends."""
        return rag.answer_from_collection(client, MODEL, fintech_collection, query)

    @tool
    def search_telecom_report(query: str) -> str:
        """Search the telecom industry report for information about mobile/broadband connectivity,
        5G, digital development, and telecommunications market trends."""
        return rag.answer_from_collection(client, MODEL, telecom_collection, query)

    return Agent(
        client=client,
        model=MODEL,
        instructions=INSTRUCTIONS,
        tools=[search_fintech_report, search_telecom_report],
    )


def main() -> None:
    load_dotenv()
    client = OpenAI()
    agent = build_agent(client)

    print("Sherlock Ohms agent ready. Ask a question (Ctrl+C to quit).")
    while True:
        try:
            query = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not query.strip():
            continue
        result = agent.run(query)
        print(result["answer"])


if __name__ == "__main__":
    main()
