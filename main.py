from getpass import getpass
from llama_cpp import Llama
from huggingface_hub import login
import time

#token = getpass("Ingrese el token: ")
#login(token=token)
#From Locally
F16 = "./model/Model-1.2B-F16.gguf"
Int8 = "./model/Model-1.2B-Q8_0.gguf"
llm = Llama(
    model_path = Int8
)
t0 = time.time()
# output = llm(
#     "Q: What is the circumference of the Earth? A:", # Prompt
#     max_tokens = 32,
#     stop = ["Q:", "\n"],
#     temperature = 0.9,
#     repeat_penalty = 1.3

# )
# output = llm.create_completion("hello", max_tokens = 32)
output  = llm.create_chat_completion(
    messages = [
        # {
        #     "role": "system",
        #     "content": "You are an assistant who speaks only Shakespearean"        
        # },
        {
            "role" : "user",
            "content" : "Describe New York in 20 words"
        }
    ]
)
t1 = time.time()
tiempo = t1 - t0
#print("Salida del modelo: ",output["choices"][0]["text"])
# Con create_chat_completion
print("Salida del modelo: ",output["choices"][0]["message"]["content"])
print("\nDemora: ", tiempo)
# From hugging_face
# llm = Llama.from_pretrained(
#     repo_id= "meta-llama/Llama-3.2-1B",
#     filename = "model.safetensors",
#     local_dir="G:/DataCamp/Working with Llama 3/Code/model"
    
# )
