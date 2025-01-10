from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import StrOutputParser 
import pprint

# Define the model path
# model_path = "./model/Model-1.2B-Q8_0.gguf"

# llm = LlamaCpp(
#     model_path=model_path,
#     temperature=0.6,
#     max_tokens=30,
#     top_p=0.4,    
#     verbose=True
# )

few_shot_inputs = [
    "La edad de Jose Carlos es 35 años",
    "Pepe cumplió 22 años hace poco",
    "Esa mujer que se llama María camina bien porque tiene 50 años",
    "Claro que Pedro el policía tiene experiencia si tiene 45 años",
    "El negro Adrián que tiene 34 años, cumplió sentencia ayer",
    "El panadero Juan Carlos tiene 73 años de vida",
    "La de Recursos Humanos (Ariana) tiene cumplió la semana pasada 63 años",
    "En la carnicería de mi casa vive Kevin y su edad es de 25 años",
    "El jefe de la policía, Raúl es amigo de mi papa y ayer festejó sus 60 años"

]
few_shot_outputs = [
    '{"nombre" : "Jose Carlos", "edad": "35"}',
    '{"nombre" : "Pepe", "edad" : "22"}',
    '{"nombre" : "María", "edad" : "50"}',
    '{"nombre" : "Pedro", "edad" : "45"}',
    '{"nombre" : "Adrián", "edad" : "34"}',
    '{"nombre" : "Juan Carlos", "edad" : "73"}',
    '{"nombre" : "Ariana", "edad" : "63"}',
    '{"nombre" : "Kevin", "edad" : "25"}',
    '{"nombre" : "Raúl", "edad" : "60"}'
]



examples = [
    {"input":few_shot_inputs[0], "output": few_shot_outputs[0]},
    {"input":few_shot_inputs[1], "output": few_shot_outputs[1]},
    {"input":few_shot_inputs[2], "output": few_shot_outputs[2]},
    {"input":few_shot_inputs[3], "output": few_shot_outputs[3]},
    {"input":few_shot_inputs[4], "output": few_shot_outputs[4]},   
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

# example_prompt = ChatPromptTemplate(
#     [
#         ("human", "{input}"),
#         ("ai", "{output}")     
#     ]
# )

# print("Few shot", few_shot_prompt.invoke({}).to_messages)
# pprint.pprint(few_shot_prompt.invoke({}).to_messages)

main_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un experto extrayendo nombres y edades de los textos y me vas a devolver un json con el nombre y la edad. No generes más token luego del JSON"),
        few_shot_prompt,
        ("human", "{input}")
    ]
)
# print("Main prompt", main_prompt)
pprint.pprint(main_prompt)
# parser = StrOutputParser()
# chain = main_prompt | llm | parser

# print(chain.invoke({"input": "Hector el jefe del equipo de análisis de datos tiene 30 años"}))