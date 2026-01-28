# Memory support:

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import MessagesState



def format_conversation_memory():
    """Format stored conversation into text for LLM prompts."""
    history = ""
    for msg in conversation_memory:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant" if isinstance(msg, AIMessage) else "System"
        history += f"{role}: {msg.content}\n"
    return history.strip()




# =========================
# Memory State
# =========================

conversation_memory = []

def add_to_memory(role, content):
    """Helper to add a message to conversation memory."""
    global conversation_memory
    if role == "user":
        conversation_memory.append(HumanMessage(content=content))
    elif role == "system":
        conversation_memory.append(SystemMessage(content=content))
    elif role == "ai":
        conversation_memory.append(AIMessage(content=content))
    
    return conversation_memory