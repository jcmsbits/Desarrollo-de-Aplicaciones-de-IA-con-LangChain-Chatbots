#from langchain.memory import ChatMessageHistory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate)
from langchain.chains import RetrievalQA