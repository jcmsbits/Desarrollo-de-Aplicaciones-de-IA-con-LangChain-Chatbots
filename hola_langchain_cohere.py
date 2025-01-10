# Esta deprecated
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import LlamaCpp
from langchain_community.llms import VLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import retrieval_qa
from langchain_community.embeddings import GPT4AllEmbeddings, LlamaCppEmbeddings
# Esta deprecated
# from langchain.vectorstores import Chroma
import requests
model_path = "./model/Model-1.2B-Q8_0.gguf"
# gpt4all_embd = GPT4AllEmbeddings()
# llama_embedding = LlamaCppEmbeddings(model_path=model_path)
# llm = LlamaCpp(
#     model_path=model_path,
#     temperature=1,
#     max_tokens=50,
#     top_p=0.4,    
#     verbose=True
# )
llm = VLLM(
    model=model_path,
    # trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
)

# Carga de documentos
text = "This is a test document."
# doc_result = gpt4all_embd.embed_documents([text])
# print(doc_result)
urls = [
    "https://arxiv.org/pdf/2306.06031v1.pdf",
    "https://arxiv.org/pdf/2306.12156v1.pdf",
    "https://arxiv.org/pdf/2306.14289v1.pdf",
    "https://arxiv.org/pdf/2306.10973v1.pdf",
    "https://arxiv.org/pdf/2306.13643v1.pdf"
]

ml_papers = []

for i, url in enumerate(urls):
    # response = requests.get(url)
    filename = f"./pdfs/paper{i+1}.pdf"
    with open(filename, "rb") as f:
        # f.write(response.content)
        # print(f"Descargado {filename}")

        loader = PyPDFLoader(filename)
        data = loader.load()
        ml_papers.extend(data)

# Utiliza la lista ml_papers para acceder a los elementos de todos los documentos descargados
# print("Contenido de ml_papers\n")
# print(type(ml_papers), len(ml_papers), type(ml_papers[3]),ml_papers[3])

# Chunk Size de 1500 caracteres
# length_function con len para medir el tipo de longitud
# chunk_overlap son 200 caracteres que van a estar en el texto anterior y 200 en el que le sigue
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1500,
#     chunk_overlap = 200,
#     length_function = len,
# )

#documents = text_splitter.split_documents(ml_papers)

#print("Longitud: ",len(documents),type(documents[10]),"Documentos: ", documents[10])

# embed_documents = llama_embedding.embed_documents([documents[0].page_content])

# vectorstore = Chroma.from_documents(
#     documents = [documents[0]],
#     embedding= llama_embedding
# )
# vectorstore._persist_directory("./db")

# retriever = vectorstore.as_retriever(
#     search_kwargs = {"k" : 3}
# )


# qa = retrieval_qa.from_chain_type(
#     llm = llm,
#     chain_type = "stuff",
#     retriever = retriever
# )

# text = "This is a test document."
# # query_result = llama.embed_query(text)
# doc_result = llama.embed_documents(documents)
# print("Embedding: ", doc_result)

# print(llm.invoke("Who you are?!"))

print(llm.invoke("What is the capital of France ?"))