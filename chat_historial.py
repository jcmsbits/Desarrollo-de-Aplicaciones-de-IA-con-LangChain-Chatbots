from langchain_community.llms import LlamaCpp
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# Define the model path
model_path = "./model/Model-1.2B-Q8_0.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.6,
    max_tokens=50,
    top_p=0.4,    
    verbose=True
)

chat_history = []

if not chat_history:
    system_message = SystemMessage(content = "Eres un asistente útil")
    chat_history.append(system_message)

query = input("Haz una pregunta: ")
chat_history.append(HumanMessage(content=query))

response = llm.invoke(chat_history)
chat_history.append(AIMessage(content = response))

# Segunda intereaccion con el usuario
query = input("Haz una pregunta: ")
chat_history.append(HumanMessage(content=query))

response = llm.invoke(chat_history)
chat_history.append(AIMessage(content = response))

# Tercera intereaccion con el usuario
query = input("Haz una pregunta: ")
chat_history.append(HumanMessage(content=query))

response = llm.invoke(chat_history)
chat_history.append(AIMessage(content = response))

print("Historial: ", chat_history)