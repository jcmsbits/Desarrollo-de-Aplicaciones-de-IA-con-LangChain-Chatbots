from langchain_community.llms import VLLM

import os
model_path = "./model/Model-1.2B-Q8_0.gguf"

llm = VLLM(
    model=model_path,
    # trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
)

print(llm.invoke("What is the capital of France ?"))