from typing import Tuple
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.openai import OpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document

from populate_database import split_documents, load_documents, clear_database, add_to_chroma

from halo import Halo
from templates import QUERY_PROMPT, RESPONSE_PROMPT, generate_history, generate_history_agent_helpers

import asyncio
import argparse
import re

from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

CHROMA_PATH = "chroma"
MODEL_NAME = "llama3"
HISTORY: list[dict[str, str]] = []


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        progress = Halo(text="Clearing Database...", spinner="dots")
        progress.start()
        clear_database()
        progress.succeed()

    progress = Halo(text="Gathering documents...", spinner="dots")
    progress.start()
    documents = load_documents()
    chunks = split_documents(documents)
    progress.succeed()
    add_to_chroma(chunks)

    model = Ollama(model=MODEL_NAME, temperature=0)
    running = True
    print("Type /exit to quit.")
    while running:
        try:
            query = input(">>> ").strip()

            
            if query.startswith("/"):
                if query[1:] == "exit": running = False
                else: print(f"Command '{query[1:]}' not found")
            elif query == "": continue
            elif not re.compile(r"^[^<>/{}[\]~`]*$").match(query):
                print("Invalid query")
                continue
            else:
                transformed_query = transform_input_to_query(query)
                if transformed_query["is_error"]: continue
                current_query: str = transformed_query["search_query"]

                search_result = document_search(current_query)
                await generate_response(transformed_query["search_query"], model, search_result["results"])
        except KeyboardInterrupt:
            print()
            continue
        except Exception as e:
            print()
            print(f"Error: {repr(e)}")
            continue

def document_search(query: str):
    progress = Halo(text=f"Searching on Chroma database for '{query}' ...", spinner='dots')
    try:
        progress.start()

        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        results = db.similarity_search_with_score(query, k=5)
        progress.succeed()
        print(results)
        return { "query": query, "results": results, "error": False }
    except Exception as e:
        progress.fail(f"An error occurred. {repr(e)}")
        return { "query": query, "results": [], "error": True }

def transform_input_to_query(user_input: str):
    """
    Transform user question to web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended search query
    """
    progress = Halo(text='Getting your input ready for search...', spinner='dots')
    try:
        progress.start()
        model_json = Ollama(model=MODEL_NAME, format='json', temperature=0)
        history_for_agents = generate_history_agent_helpers(HISTORY)
        query_chain = QUERY_PROMPT | model_json | JsonOutputParser()

        gen_query = query_chain.invoke({"history": history_for_agents, "input": user_input})
        search_query = gen_query["query"]
        progress.succeed()
        return {"search_query": search_query, "is_error": False}
    except Exception as e:
        progress.fail(f"An error occurred. {repr(e)}")
        return { "search_query": None, "is_error": True }

async def generate_response(query: str, model: Ollama, results: list[Tuple[Document, float]] = list()):
    global HISTORY
    print()
    chain = RESPONSE_PROMPT | model | StrOutputParser()
    history = generate_history(HISTORY)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) if len(results) > 0 else "No contexts found"
    sources = [f"{_score:.2f}\t" + doc.metadata.get("id", None) for doc, _score in results]

    data_to_pass = { "history": history, "context": context_text, "message": query }

    text = ""
    for chunk in chain.stream(data_to_pass):
        print(chunk, end="", flush=True)
        text += chunk
    print("\n")
    if len(sources) > 0:
        print("-------------------------------")
        print(f"Score\tSources:\n{"\n".join(sources)}")
        print("-------------------------------")


    HISTORY += [
        {
            "entity": "user",
            "message": query,
            "context": context_text
        },
        {
            "entity": "assistant",
            "message": text,
        }
    ]
    return text


if __name__ == "__main__":
    asyncio.run(main())