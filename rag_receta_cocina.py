from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import  create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
import multiprocessing
from pprint import pprint

local_model = "./model/Model-1.2B-Q8_0.gguf"
gpt4all_embd = FastEmbedEmbeddings()

loader = DirectoryLoader(
    "./archivos",
    glob= "**/*.pdf",
    loader_cls= PyPDFLoader
)
pages = loader.load()

# pprint(pages)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 60
)

splits = text_splitter.split_documents(pages)

# pprint(splits)

# pprint(splits[100]) 

vectorstore = Chroma.from_documents(
    documents= splits,
    embedding= gpt4all_embd,
    persist_directory= "./bd"
)

pprint(f"Vectorstore:  {vectorstore}")

retriever = vectorstore.as_retriever() 
pprint(f"Retriever: {retriever}")
system_prompt = (
    "Eres un asistente que devuelve información de múltiples PDFs, además incluye emojis a cada una de las respuestas, tienes el siguiente {context}"    
    )

llm = ChatLlamaCpp(
    temperature=1,
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

contextualize_q_system_prompt = (
    "Responde según el historial de chat y la última pregunta del usuario \
    Si no está en el historial de chat o en el contexto. No respondas la pregunta \
    Además responde de manera profesional a la pregunta del usuario"
)

contextualize_q_system_ = ChatPromptTemplate.from_messages(
    [
        (
            "system", contextualize_q_system_prompt
        ),
        # MessagesPlaceholder("context"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
# pprint(f"Contextualize: {contextualize_q_system_}")

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_system_
)
# prompt = "Como calentar agua en el sartén para que alcance de ebullición?"

# conversation = [HumanMessage("La comida es muy buena en este restaurante"), AIMessage("Si, muchos clientes han manifestado lo mismo")]

# template = ChatPromptTemplate.from_messages(
#     [       
#         ("human", "La comida es muy buena en este restaurante"),
#         ("ai", "Si, muchos clientes han manifestado lo mismo"),
#         # ("human", "{input}"),
#     ]
# )
# template2 = template.partial(user="Lucy", name="R2D2")
# template3 = template2.format_messages(input="hello")

# print(history_aware_retriever.invoke({"input": prompt, "context" : template3}))

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
# pprint(f"qa_prompt: {qa_prompt}")

question_answer_question = create_stuff_documents_chain(
    llm, qa_prompt
)

rag_chain = create_retrieval_chain(  
        history_aware_retriever,      
        question_answer_question,
           
    )
print(rag_chain)
store = {}

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key ='input',
    # history_messages_key = 'chat_history'
    history_messages_key = 'chat_history',
    output_messages_key = 'answer'
)
# pprint(f"conversational_rag_chain {conversational_rag_chain}")

print(conversational_rag_chain.invoke(
    {
        'input' : 'Cocinar con las verduras con pollo?'
    },
    config = {
        'configurable' : {"session_id" : 'abc123'}
    }
))
# ['answer']