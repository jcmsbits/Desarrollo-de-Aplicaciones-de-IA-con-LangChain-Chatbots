from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import LlamaCpp
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import (BaseChatMessageHistory, InMemoryChatMessageHistory)
from langchain_core.runnables.history import RunnableWithMessageHistory

# Define the model path
#model_path = "./model/Model-1.2B-Q8_0.gguf"
model_path = "./model/Model-1.2B-F16.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.8,
    max_tokens=250,
    top_p=0.6,    
    verbose=True
)
store = {}
def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
     [
         ("system", "You are a helpful assistant. Answer all question to the best of your ability"),
         MessagesPlaceholder(variable_name= "messages")
     ]
)

chain = prompt | llm

response = chain.invoke(
    {"messages" : [HumanMessage(content = "Hi! I am Bob")]}
    )
print(response)

response = chain.invoke(
    {"messages" : [HumanMessage(content = "What is my name?")]}
    )
print(response)

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable" : {"session_id" : "abc5"}}

response = with_message_history.invoke(
    [HumanMessage(content="What is name name")],
    config = config
)