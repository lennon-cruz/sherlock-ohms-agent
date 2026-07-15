"""Retrieve-Augment-Generate: turn retrieved chunks into a grounded answer."""

from openai import OpenAI

from sherlock_ohms import vectorstore

SYSTEM_PROMPT = "You are an assistant for question-answering tasks."

PROMPT_TEMPLATE = (
    "Use the following pieces of retrieved context to answer the question. "
    "If the answer isn't contained in the context, say you don't know.\n"
    "\n# Question:\n{question}\n"
    "\n# Context:\n{context}\n"
    "\n# Answer:"
)


def answer_from_collection(client: OpenAI, model: str, collection, question: str) -> str:
    documents = vectorstore.query(collection, question)
    context = "\n\n".join(documents) if documents else "(no matching context found)"

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(question=question, context=context)},
        ],
    )
    return response.choices[0].message.content
