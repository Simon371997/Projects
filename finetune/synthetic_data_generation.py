import json
from typing import List
from pydantic import BaseModel

from colorama import Fore

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from litellm import completion
from generated_prompt import prompt_template


class Record(BaseModel):
    question: str
    answer: str


class Response(BaseModel):
    generated: List[Record]


def llm_call(data: str, num_records: int = 5) -> dict:
    stream = completion(
        model="ollama/qwen3:latest",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        options={"num_predict": 2000},
        format=Response.model_json_schema(),
        api_base="http://localhost:11434",
    )
    data = ""
    for x in stream:
        delta = x["choices"][0]["delta"]["content"]
        if delta is not None:
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            data += delta
    return json.loads(data)


if __name__ == "__main__":
    converter = DocumentConverter()
    document = converter.convert("echa_overview.pdf").document
    chunker = HybridChunker()

    # Chunk the document
    chunks = chunker.chunk(dl_doc=document)

    dataset = {}
    for i, chunk in enumerate(chunks):
        print(Fore.YELLOW + f"Raw Text: \n{chunk.text[:300]}..." + Fore.RESET)
        enriched_text = chunker.contextualize(chunk=chunk)
        print(
            Fore.LIGHTMAGENTA_EX
            + f"Contextualized Text: \n{enriched_text[:300]}..."
            + Fore.RESET
        )

        data = llm_call(enriched_text)
        dataset[i] = {"generated": data["generated"], "context": enriched_text}

    with open("tm1dataset.json", "w") as f:
        json.dump(dataset, f)
