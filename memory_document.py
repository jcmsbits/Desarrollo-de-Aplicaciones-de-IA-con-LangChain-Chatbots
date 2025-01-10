from langchain_community.llms import LlamaCpp
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from langchain_core.chat_history import (BaseChatMessageHistory, InMemoryChatMessageHistory)
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)

#Define the model path
model_path = "./model/Model-1.2B-Q8_0.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.8,
    max_tokens=250,
    top_p=0.6,    
    verbose=True
)




class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


# history = get_by_session_id("1")
# history.add_message(AIMessage(content="hello"))
# print(store)  # noqa: T201


prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at question"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    # Uses the get_by_session_id function defined in the example
    # above.
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

print(chain_with_history.invoke(  # noqa: T201
    {"ability": "math", "question": "What does cosine mean?"},
    config={"configurable": {"session_id": "foo"}}
))

# Uses the store defined in the example above.
print(store)  # noqa: T201

print(chain_with_history.invoke(  # noqa: T201
    {"ability": "math", "question": "What's its inverse"},
    config={"configurable": {"session_id": "foo"}}
))

print(store)  # noqa: T201


# config = {"configurable" : {"session_id": "abc2"}}

# response_1 = chain_with_history.invoke(
#     {"question": [HumanMessage(content = "HI! I'm Bob")]},
#     config= config
# )
# print("Respueta 1:", response_1)

# response_2 = chain_with_history.invoke(
#      {"question": [HumanMessage(content = "What is my name?")]},
#     config = config
# )
# print("Respueta 2:", response_2)
# print("Store: ", store)