from typing import Tuple, List
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.openai import OpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document

from utils import split_documents, load_documents, clear_database, add_to_chroma

from halo import Halo
from templates import ROUTING_PROMPT, QUERY_PROMPT, RESPONSE_PROMPT, generate_history

import asyncio
import argparse
import re

from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

CHROMA_PATH = "chroma"
MODEL_NAME = "mistral"
HISTORY: list[dict[str, str]] = []


async def main():
    """The main progam"""
    global HISTORY
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
                HISTORY += [
                    {
                        "entity": "user",
                        "message": query
                    }
                ]

                error = False
                results: List[Tuple[Document, float]] = []
                instructions = route()
                for instruction in instructions:
                    if instruction["choice"] == "document_search":
                        agent_query, error = transform_input_to_query(instruction["suggestion"])
                        if error: break
                        search_result, error = document_search(agent_query)
                        if error: break
                        results += search_result
                    if instruction["choice"] == "generate":
                        await generate_response(model, instruction["suggestion"], results)
        except KeyboardInterrupt:
            print()
            continue
        except Exception as e:
            print()
            print(f"Error: {repr(e)}")
            continue

def route():
    """Routes the conversation"""
    global HISTORY
    progress = Halo(text=f"Just a moment...", spinner='dots')
    try:
        progress.start()

        model_json = Ollama(model=MODEL_NAME, format='json', temperature=0)
        history_for_agents = generate_history(HISTORY)
        route_chain = ROUTING_PROMPT | model_json | JsonOutputParser()
        data = route_chain.invoke({ "history": history_for_agents })
        instructions : list[dict[str, str]] = data["instructions"]
        progress.succeed()
        return instructions
    except Exception as e:
        progress.fail(f"An error occurred with Routing Agent: '{repr(e)}'.")
        return [{ "generate": "Tell the user that you cannot find the answer to the user's input/question." }]
        

def document_search(query: str):
    """Search for information on Chroma Database"""
    global HISTORY
    progress = Halo(text=f"Searching on Chroma database for '{query}' ...", spinner='dots')
    try:
        progress.start()

        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        results = db.similarity_search_with_score(query, k=5)

        result_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) if len(results) > 0 else "No contexts found"
        HISTORY += [
            {
                "entity": "query",
                "input": query,
                "result": result_text
            }
        ]
        progress.succeed(f"Searched on Chroma database for '{query}'. Found {len(results)} results.")
        return results, False
    except Exception as e:
        progress.fail(f"An error occurred. {repr(e)}")
        return [], True

def transform_input_to_query(suggestion: str):
    """Transforms the user question to web search"""
    global HISTORY
    try:
        model_json = Ollama(model=MODEL_NAME, format='json', temperature=0)
        history_for_agents = generate_history(HISTORY)
        query_chain = QUERY_PROMPT | model_json | JsonOutputParser()

        gen_query = query_chain.invoke({ "history": history_for_agents, "suggestion": suggestion })
        search_query: str = gen_query["query"]
        return search_query, False
    except Exception as e:
        return "", True

async def generate_response(model: Ollama, suggestion: str, results: list[Tuple[Document, float]] = list()):
    global HISTORY
    print()
    chain = RESPONSE_PROMPT | model | StrOutputParser()
    history = generate_history(HISTORY)

    sources = [f"{_score:.2f}\t" + doc.metadata.get("id", None) for doc, _score in results]

    text = ""
    for chunk in chain.stream({ "history": history, "suggestion": suggestion }):
        print(chunk, end="", flush=True)
        text += chunk
    print("\n")
    if len(sources) > 0:
        print("-------------------------------")
        print(f"Score\tSources:\n{'\n'.join(sources)}")
        print("-------------------------------")
    return text


if __name__ == "__main__":
    asyncio.run(main())