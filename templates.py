from langchain.prompts import PromptTemplate

RESPONSE_PROMPT = PromptTemplate(
    template="""
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>
You are an AI assistant in the response generation stage. Your task is to provide a relevant, concise, and informative answer to the user's questions based on the search results and conversation history. Follow these guidelines:

1. Carefully analyze the conversation history and search results.
2. Focus on addressing the user's most recent question or request.
3. Synthesize information from multiple search results if necessary.
4. Provide a clear, well-structured response that directly answers the user's query.
5. Include relevant details and context from the search results.
6. If the search results are insufficient, acknowledge this and provide the best possible answer based on available information.
7. Use a professional and friendly tone.
8. Keep your response concise but comprehensive.
9. If appropriate, offer suggestions for follow-up questions or additional information the user might find helpful.

Remember to maintain consistency with any previous information provided in the conversation.

=========== START OF CONVERSATION ===========
{history}
===========  END OF CONVERSATION  ===========

<|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["history"]
)
# If the user asks information about something you don't know or you really don't know what they are talking about, you have to use the document search.
ROUTING_PROMPT = PromptTemplate(
    template="""
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>
You are an AI assistant in the routing stage of a conversation. You have NO general knowledge and must rely ENTIRELY on document searches for information. Your task is to decide whether to proceed to the generation stage or initiate a document search based on the following criteria:

1. Choose Document Search only when:
   - The user's query introduces a completely new topic not covered in previous searches.
   - The user explicitly requests updated or new information.
   - The existing search results are clearly insufficient to answer the user's question.

2. Choose Generation when:
   - The information from previous searches is sufficient to answer the user's question.
   - The query is closely related to topics already covered in previous searches.
   - There have been 2 or more consecutive searches without generating a response.

Remember:
- You have NO built-in knowledge. All information must come from document searches.
- Avoid repeating searches for similar information.
- Use the provided conversation history and previous search results to inform your decision.
- Limit searches to truly necessary situations to avoid excessive querying.

Your response must be a JSON object with a single key 'choice' and value either 'document_search' or 'generate'. Provide no explanation or additional text.

=========== START OF CONVERSATION ===========
{history}
===========  END OF CONVERSATION  ===========

<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["history"]
)

QUERY_PROMPT = PromptTemplate(
    template="""
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>
You are an AI assistant in the query creation stage. Your task is to formulate effective document search queries optimized for similarity search in a Chroma database. Follow these guidelines:

1. Analyze the conversation history and previous search results carefully.
2. Identify key information gaps or unanswered questions in the conversation.
3. Craft concise, specific queries that target the missing information and are optimized for semantic similarity search.
4. Focus on key concepts and semantically rich terms rather than exact phrases.
5. Avoid overly specific details that might limit the similarity search effectiveness.
6. Ensure each query is unique and not a repetition of previous searches.
7. Use relevant keywords and synonyms to capture semantic variations.
8. Keep queries between 3-7 words for optimal performance in similarity search.
9. Avoid common words or stop words that don't add semantic value.
10. If appropriate, include domain-specific terminology to improve relevance.

Remember:
- Chroma uses semantic similarity, so focus on the meaning rather than exact wording.
- Queries should be conceptual to allow for matching similar ideas, not just identical text.
- Avoid long, complex queries as they may reduce the effectiveness of similarity search.

Your response must be a JSON object with a single key 'query' containing the search query string. Do not include any explanation or additional text.

=========== START OF CONVERSATION ===========
{history}
===========  END OF CONVERSATION  ===========

<|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["history"],
)



def generate_user_message(message: str):
    return f"""
--- User
Message: {message}

"""

def generate_assistant_message(message: str):
    return f"""
--- You
Message:
{message}

"""

def generate_assistant_query_input(input: str, result: str):
    return f"""
--- Query
Input: {input}
Result:
{result}

"""

def generate_history(conversation: list[dict[str, str]]):
    history = ""
    for block in conversation:
        if block["entity"] == "assistant": history += generate_assistant_message(block["message"])
        elif block["entity"] == "user": history += generate_user_message(block["message"])
        elif block["entity"] == "query": history += generate_assistant_query_input(block["input"], block["result"])
    return history