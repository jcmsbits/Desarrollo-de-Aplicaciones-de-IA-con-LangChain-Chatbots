from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Define the model path
model_path = "./model/Model-1.2B-Q8_0.gguf"

# Set up callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Create the LLM object
print("Cargando el modelo...")
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=250,
    top_p=1.0,
    callback_manager=callback_manager,
    verbose=True
)
print("Generando el prompt...")
# Example usage
# question = "What is bindings in programming languages?"
# question = "Haciendo una historia sobre Cuba"
question = [
    (
        "system",
        "Eres un profesor de programación en python. Enseña a programar"     
     ),
     (
         'human',
         #"Quiero instalar python en windows"
         "Explicame como instalar python en windows"
     )
]
question = [
    (
        "system",
        "Eres un vendedor de pizz napolitana"     
     ),
     (
         'human',
         #"Quiero instalar python en windows"
         "Que sabor tienes"
     )
]
# response = llm.invoke( question)
# print(response)
prompt_template = PromptTemplate.from_template("Dime un chiste {topic}")
print(prompt_template.invoke({"topic":"gatos"}))


chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil"),
    ("user", "Dime un chiste {topic}")
])
print(chat_prompt_template.invoke({"topic": "gatos"}))

placeholder_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un asistente útil"),
        MessagesPlaceholder("msgs")
    ]
)
print(placeholder_prompt_template.invoke({"msgs" : [HumanMessage(content = "hola"), HumanMessage(content = "adios")]}))