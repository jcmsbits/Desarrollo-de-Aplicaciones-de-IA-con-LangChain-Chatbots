from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant named {name}."),
        ("human", "Hi I'm {user}"),
        ("ai", "Hi there, {user}, I'm {name}."),
        ("human", "{input}"),
    ]
)
pprint(template)
template2 = template.partial(user="Lucy", name="R2D2")
pprint(template2)

template3 = template2.format_messages(input="hello")
pprint(template3)

pprint(template2.pretty_print())

template.save("./")