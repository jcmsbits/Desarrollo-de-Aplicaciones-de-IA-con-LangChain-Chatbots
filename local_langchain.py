from langchain_community.llms import LlamaCpp
print("Cargando el modelo...")
llm = LlamaCpp(model_path="./model/Model-1.2B-Q8_0.gguf")
print("Generando intrucción...")
llm.invoke("Simulate a rap battle between Stephen Colbert and John Oliver"
           )