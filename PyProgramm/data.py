import os
import torch
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import xml.etree.ElementTree as ET


# Sentence Transformer initalisieren
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(prefixed, convert_to_numpy=True, batch_size=32, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"query: {text}"
        embedding = self.model.encode([prefixed], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0].tolist()


# PDF-Parsing
def process_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)

            # PyMuPDFLoader
            pdf_loader = PyMuPDFLoader(file_path)
            pdf_documents = pdf_loader.load()

            # Extrahierte Dokumente zur Liste hinzufügen
            documents.extend(pdf_documents)

    return documents


# XML Parsen
# Namespace-Deklarationen extrahieren
namespaces = {
    "n": "http://www.schema.de/2004/ST4/XmlImportExport/Node",
    "d": "http://www.schema.de/2004/ST4/XmlImportExport/Data",
    "l": "http://www.schema.de/2004/ST4/XmlImportExport/Link",
    "m": "http://www.schema.de/2004/ST4/XmlImportExport/Meta"
}

# Funktion: Rekursive Extraktion von Inhalten
def extract_content(element):
    content = []
    if element.text and element.text.strip():
        content.append(element.text.strip())
    for child in element:
        content.append(extract_content(child))
    if element.tail and element.tail.strip():
        content.append(element.tail.strip())
    return "".join(content)

# Funktion: Deutsche Inhalte extrahieren
def extract_deutsch_content(root):
    german_content = []
    for value in root.findall(".//n:Value[@n:Aspect='de']", namespaces):
        entries = value.findall(".//n:Entry", namespaces)
        for entry in entries:
            content = extract_content(entry)
            german_content.append(content)
    return german_content

# Funktion: Bereinigen leere Absätze
def remove_redundant_blank_lines(file_content):
    lines = file_content.splitlines()
    cleaned_lines = []
    blank_count = 0

    for line in lines:
        if not line.strip():
            blank_count += 1
        else:
            blank_count = 0  

        if blank_count <= 2:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

# XML-Dateien verarbeiten und zu documents hinzufügen
def process_xmls(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Deutsche Inhalte extrahieren
            german_content = extract_deutsch_content(root)
            combined_content = "\n\n".join(german_content)

            # Bereinigung anwenden
            cleaned_content = remove_redundant_blank_lines(combined_content)

            # In Document-Objekt umwandeln
            documents.append(Document(page_content=cleaned_content, metadata={"source": filename}))
    return documents


# Hauptverzeichnis
data_directory = r"C:\Users\fabio.cappellaro\Documents\Masterarbeit Projekt\Masterarbeit_FC\Datenpool"

# XML-Dokumente verarbeiten
xml_documents = process_xmls(data_directory)

# PDF-Dokumente verarbeiten
pdf_documents = process_pdfs(data_directory)

# Alle Dokumente kombinieren
documents = xml_documents + pdf_documents

# Text in Chunks splitten
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=20,
    length_function=len
)
docs = text_splitter.split_documents(documents)


# Embedding-Modell laden und in Vektorspeicher speichern
embedding_model = "intfloat/multilingual-e5-large"
embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)

vectorstore = FAISS.from_documents(docs, embeddings)

print(f"{len(docs)} Dokument-Chunks wurden verarbeitet und in den Vektorspeicher geladen.")


# Debugging

# Funktion zum Exportieren von Chunks, Embeddings und Text
def export_data(docs, vectorstore, directory):
    os.makedirs(directory, exist_ok=True)

    # Originaltext exportieren
    text_path = os.path.join(directory, "parsed_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.page_content + "\n\n")

    # Chunks exportieren
    chunks_path = os.path.join(directory, "chunks.txt")
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write("Anzahl Chunks: " + str(len(docs)) + "\n\n")
        for doc in docs:
            f.write(f"[{doc.page_content}]\n\n")

    # Embeddings exportieren
    embeddings_path = os.path.join(directory, "embeddings.txt")
    with open(embeddings_path, "w", encoding="utf-8") as f:
        f.write("Anzahl Embeddings: " + str(vectorstore.index.ntotal)+ "\n\n")
        for doc_id, embedding in enumerate(vectorstore.index.reconstruct_n(0, len(docs))):
            f.write(f"Document {doc_id}: {embedding}\n")

    print(f"Export abgeschlossen. Dateien gespeichert in: {directory}")

# Daten exportieren
export_directory = r"C:\\Users\\fabio.cappellaro\\Documents\\Masterarbeit_FC\\ExportedData"
export_data(docs, vectorstore, export_directory)