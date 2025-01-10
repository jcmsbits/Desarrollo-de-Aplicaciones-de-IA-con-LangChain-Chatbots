from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("./archivos/example - 1.txt") as f:
    state_of_the_union = f.read()

text_splitter_1 = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap =20,
    length_function = len,
    is_separator_regex=False
)

text_splitter_2 = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap =2,  
    length_function = len,
    is_separator_regex=False
)

text_1 =text_splitter_1.create_documents([state_of_the_union])
print(text_1)


text_2 = text_splitter_2.create_documents([state_of_the_union])
print(text_2)