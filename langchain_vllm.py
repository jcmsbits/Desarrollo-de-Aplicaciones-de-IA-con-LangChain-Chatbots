from langchain_community.llms import VLLM
from langchain_community.chat_models import ChatLlamaCpp
import multiprocessing

local_model = "./model/Model-1.2B-Q8_0.gguf"

# llm = ChatLlamaCpp(
#     temperature=1,
#     model_path=local_model,
#     n_ctx=2512,
#     # n_gpu_layers=8,
#     n_batch=5,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#     max_tokens=1024,
#     n_threads=multiprocessing.cpu_count() - 1,
#     repeat_penalty=1.5,
#     top_p=0.5,
#     verbose=True,
# )
llm =  VLLM(
    model=local_model,
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,

)

print(llm.invoke("What is the capital of France ?"))