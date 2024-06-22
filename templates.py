from langchain.prompts import PromptTemplate

RESPONSE_PROMPT = PromptTemplate(
    template="""  
You are an AI assistant for Private Document Search, that can answer questions from the user, either having document context or not.
If you have conversed with the user before, you will be given previous conversations with you and the user to help you answer.
Strictly use the following pieces of documents provided by other AI Agents to answer the question. If you don't know the answer, just say that you don't know.
Keep the answer concise and accurate, but provide all of the details you can. 
Only make direct references to material that answers the user's question if provided in the context.
If the document context explicitly states that no context is needed, you can straightforwardly answer the user.
    
=========== START OF CONVERSATION ===========
{history}
---   User
Message:
{message}

--- AI Agent
Document Context:
{context} 

---   You
Message:
""",
    input_variables=["history", "message", "context"]
)
# If the user asks information about something you don't know or you really don't know what they are talking about, you have to use the document search.
ROUTING_PROMPT = PromptTemplate(
    template="""
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>
You are an expert at helping an AI Assistant to route a user input to either the generation stage or document search.
You will be given previous conversations between the AI Assistant and the user to help you understand the situation.
You can use the document search for any key information in the database. If you do not understand anything, you must proceed with document search.
Otherwise, you can skip and go straight to the generation phase to respond.
You do not need to be stringent with the keywords in the input related to these topics.
Give a binary choice 'document_search' or 'generate' based on the input.
Return the JSON with a single key 'choice' with no premable or explanation.

=========== START OF CONVERSATION ===========
{history}
===========  END OF CONVERSATION  ===========

Question to route: {input}

<|eot_id|>
    
<|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["history", "input"]
)

QUERY_PROMPT = PromptTemplate(
    template="""
<|begin_of_text|>
    
<|start_header_id|>system<|end_header_id|> 
    
You are an expert at helping an AI Assistant to craft document search queries for Private Document Search.
You will be given previous conversations between the AI Assistant and the user to help you understand the situation.
More often than not, a user will input that they wish to learn more about, however it might not be in the best format. 
Reword their input to be the most effective search string possible.
Return the JSON with a single key 'query' with no premable or explanation.

=========== START OF CONVERSATION ===========
{history}
===========  END OF CONVERSATION  ===========
    
Input to transform:
{input} 
    
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>    
""",
    input_variables=["history", "input"],
)

def generate_user_message(message: str, context: str):
    return f"""
---   User
Message:
{message}

--- AI Agent
Document Context:
{context}

"""

def generate_assistant_message(message: str, is_assistant = True):
    return f"""
---  {'You' if is_assistant else 'AI Assistant'}
Message:
{message}

"""

def generate_history(conversation: list[dict[str, str]]):
    history = ""
    for block in conversation:
        if block["entity"] == "assistant": history += generate_assistant_message(block["message"])
        elif block["entity"] == "user": history += generate_user_message(block["message"], block["context"])
    return history

def generate_history_agent_helpers(conversation: list[dict[str, str]]):
    history = ""
    for block in conversation:
        if block["entity"] == "assistant": history += generate_assistant_message(block["message"], False)
        elif block["entity"] == "user": history += generate_user_message(block["message"], block["context"])
    return history