from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain.output_parsers.json import SimpleJsonOutputParser

# Define the model path
model_path = "./model/Model-1.2B-Q8_0.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.6,
    max_tokens=50,
    top_p=0.4,    
    verbose=True
)

examples = [
    {"input":"Cuanto es 2 + 2", "output": "4"},
    {"input":"Cuanto es 3 + 2", "output": "5"},
    {"input":"Cuanto es 8 + 2", "output": "10"},
    {"input":"Cuanto es 4 + 1", "output": "5"},
    {"input":"Cuanto es 9 + 0", "output": "9"},
    {"input":"Cuanto es 7 + 4", "output": "11"},
    {"input":"Cuanto es 6 + 1", "output": "7"},
]

example_prompt = ChatPromptTemplate(
    [
        ("human", "{input}"),
        ("ai", "{output}")     
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples = examples
)
print("Few shot", few_shot_prompt.invoke({}).to_messages)

main_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un experto en matematicas"),
        few_shot_prompt,
        ("human", "{input}")
    ]
)
print("Main prompt", main_prompt)

chain = main_prompt | llm

print(chain.invoke({"input": "Cuanto es 2 + 9"}))


chat_bot_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Traduce lo siguiente al {language}:"),
        ("human", "{text}")
    ]
)

parser = StrOutputParser()
# parser= SimpleJsonOutputParser()

chain = chat_bot_prompt_template | llm | parser

response = chain.stream(
                        {
                        "language": "Italian", 
                         "text": "Yo soy un cubano que baila macarena cuando quiera"
                         }
                         )
print(response)