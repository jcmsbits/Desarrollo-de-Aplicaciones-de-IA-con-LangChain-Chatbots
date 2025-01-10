from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatLlamaCpp
import multiprocessing

local_model = "./model/Model-1.2B-Q8_0.gguf"

prompt = ChatPromptTemplate.from_messages(
    [("system", "What are everyone's favorite colors:\n\n{context}")]
)
llm = ChatLlamaCpp(
    temperature=0.5,
    model_path=local_model,
    n_ctx=2512,
    # n_gpu_layers=8,
    n_batch=5,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    max_tokens=1024,
    n_threads=multiprocessing.cpu_count() - 1,
    repeat_penalty=1.5,
    top_p=0.5,
    verbose=True,
)



chain = create_stuff_documents_chain(llm, prompt)

docs = [
    Document(page_content="Jesse loves red but not yellow"),
    Document(page_content = "Jamal loves green but not as much as he loves orange")
]

print(chain.invoke({"context": docs}))