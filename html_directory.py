from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader

file_path = './archivos/arcih.html'

loader = BSHTMLLoader(file_path)
data = loader.load()

print(data)