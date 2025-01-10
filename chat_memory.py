from langchain_community.llms import LlamaCpp
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.chat_history import (BaseChatMessageHistory, InMemoryChatMessageHistory)
from langchain_core.runnables.history import RunnableWithMessageHistory


# Define the model path
model_path = "./model/Model-1.2B-Q8_0.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.8,
    max_tokens=250,
    top_p=0.6,    
    verbose=True
)
response_1 = llm.invoke([HumanMessage(content = "Hi! my name is bob")])
print(AIMessage(content = response_1))

response_2 = llm.invoke([HumanMessage(content = "What is my name?")])
print(AIMessage(content =response_2))

print(llm.invoke(
    [
        HumanMessage(content = "Hi! my name is bob"),
        AIMessage(content = " Hello Bob! How can I assist you, Bob"),
        HumanMessage(content = "What is my name?")
    ]
)
)

# store = {}

# def get_session_history(session_id:str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

# with_message_histoy = RunnableWithMessageHistory(llm, get_session_history )

# config = {"configurable" : {"session_id": "abc2"}}

# response_1 = with_message_histoy.invoke(
#     [HumanMessage(content = "HI! I'm Bob")],
#     config= config
# )
# print("Respueta 1:", response_1)

# response_2 = with_message_histoy.invoke(
#     [HumanMessage(content = "What is my name?")],
#     config = config
# )
# print("Respueta 2:", response_2)
# print("Store: ", store)

## Otro usuario
# config = {"configurable" : {"session_id": "abc3"}}

# response_3 = with_message_histoy.invoke(
#     [HumanMessage(content = "HI! I'm Bob")],
#     config= config
# )
# print("Respueta 1:", response_3)

# response_4 = with_message_histoy.invoke(
#     [HumanMessage(content = "What is my name?")],
#     config = config
# )
# print("Respueta 2:", response_4)
# print("Store: ", store)