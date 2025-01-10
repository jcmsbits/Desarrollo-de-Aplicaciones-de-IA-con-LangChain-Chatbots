from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import time

# Define the model path
model_path = "./model/Model-1.2B-Q8_0.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.6,
    max_tokens=50,
    top_p=0.4,    
    verbose=True
)

# Making sequence with Runnable
sequence = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x : x * 2)

print("Sequence: ",sequence.invoke(10))

sequence_2 = RunnableLambda(lambda x: x + 5) | {
    "index_1" : RunnableLambda(lambda x: x * 6),
    "index_2" : RunnableLambda(lambda x: x * 7)
}
print("Sequence 2: ",sequence_2.invoke(5))

# Using JSONOutputParser

joke_query = "tell me a joke"

parser  = JsonOutputParser()

prompt = PromptTemplate(
    template = "Answer the user query. With a good answer \n{format_intructions}\n{query}",
    input_variables = ["query"],
    partial_variables = {"format_intructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
# chain = prompt | llm

# response = chain.invoke({"query": joke_query})
chunks = []
for chunk in chain.stream({"query": joke_query}):
    print(chunk, end="", flush = True)
    chunks.append(chunk)


print("Respuesta JSONOutputParser: ", chunks)

# for s in chain.stream({"query": joke_query}):
#     print(s, "-")
#     time.sleep(0.3)

# Using Stream

# chunks = []
# async for chunk in llm.astream(joke_query):
#     chunks.append(chunk)
#     print(chunk.content, end = "-", flush = True)
#     time.sleep(0.3)

# print("Generando respuesta: ")
# for chunk in llm.stream(joke_query):
#     chunks.append(chunk)
#     # print(chunk, end = "-", flush = True)
#     print(chunk, end = "", flush = True)
#     # time.sleep(0.3)
# print("\nChunks:", chunks)