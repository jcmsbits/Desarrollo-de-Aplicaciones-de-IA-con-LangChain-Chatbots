from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

file_path = "./archivos"

loader = DirectoryLoader(file_path, glob = "**/*.pdf", show_progress=True)
docs = loader.load()
print(docs)

file_path_pdf = (
    "./archivos/lorem-ipsum.pdf"
    )
file_path_csv = (
    "./archivos/online_retail.csv"
)
loader = PyPDFLoader(file_path_pdf)

pages = loader.load_and_split()

print(pages[0])

loader = CSVLoader(file_path=file_path_csv)
data = loader.load()

for record in data[:2]:
    print(record)
    print(type(record))