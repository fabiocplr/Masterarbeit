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
        embeddings = self.model.encode(prefixed, convert_to_numpy=True, batch_size=32, normalize_embeddings=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"query: {text}"
        embedding = self.model.encode([prefixed], convert_to_numpy=True, normalize_embeddings=False)
        return embedding[0].tolist()


# XML Parsen
# Namespaces
namespaces = {
    "n": "http://www.schema.de/2004/ST4/XmlImportExport/Node",
    "d": "http://www.schema.de/2004/ST4/XmlImportExport/Data",
    "l": "http://www.schema.de/2004/ST4/XmlImportExport/Link",
    "m": "http://www.schema.de/2004/ST4/XmlImportExport/Meta"
}

def extract_content(element):
    #Rekursive Extraktion des Textinhalts eines Elements
    content = []
    if element.text and element.text.strip():
        content.append(element.text.strip())
    for child in element:
        content.append(extract_content(child))
    if element.tail and element.tail.strip():
        content.append(element.tail.strip())
    return " ".join(content).strip()

def extract_content(element):
    """Rekursive Extraktion des Textinhalts eines Elements."""
    content = []
    if element.text and element.text.strip():
        content.append(element.text.strip())
    for child in element:
        content.append(extract_content(child))
    if element.tail and element.tail.strip():
        content.append(element.tail.strip())
    return " ".join(content).strip()

def extract_deutsch_chunks(root):
    chunks = []
    current_chunk = ""
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]  # Namespace entfernen
        if tag == "Data-Title":
            # Für jeden Data-Title, der einen deutschen Tag hat
            for value in elem.findall("n:Value[@n:Aspect='de']", namespaces):
                texts = []
                for entry in value.findall("n:Entry", namespaces):
                    t = extract_content(entry)
                    if t:
                        texts.append(t)
                if texts:
                    # Falls bereits ein Chunk vorhanden ist, speichern und neu starten
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    current_chunk += "\n\n".join(texts) + "\n\n"
        elif tag == "Data-Content":
            # Alle Data-Content Elemente mit Aspect="de" dem aktuellen Chunk hinzufügen
            for value in elem.findall("n:Value[@n:Aspect='de']", namespaces):
                texts = []
                for entry in value.findall("n:Entry", namespaces):
                    t = extract_content(entry)
                    if t:
                        texts.append(t)
                if texts:
                    current_chunk += "\n\n".join(texts) + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Neue Funktion: Kleine XML-Chunks (< 100 Zeichen) mit dem nächsten Chunk zusammenführen
def merge_small_xml_chunks(chunks, threshold=100):
    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i].strip()
        # Solange der aktuelle Chunk unter dem Schwellenwert liegt und es einen nächsten Chunk gibt,
        # wird dieser an den aktuellen Chunk angehängt.
        while len(current) < threshold and i + 1 < len(chunks):
            i += 1
            current += " " + chunks[i].strip()
        merged.append(current)
        i += 1
    # Falls der letzte Chunk immer noch zu klein ist und es bereits einen vorherigen Chunk gibt,
    # wird der letzte Chunk an den vorangehenden Chunk angehängt.
    if merged and len(merged[-1]) < threshold and len(merged) > 1:
        merged[-2] = merged[-2] + " " + merged[-1]
        merged.pop()
    return merged


# Für jede XML-Datei wird für jeden Chunk (Data-Title + zugehörige Data-Content Element) ein Dokument erstellt, Kapitel werden als Metadatum mitgeben
def process_xmls(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()
            chunks = extract_deutsch_chunks(root)
            # Kleine XML-Chunks zusammenführen
            chunks = merge_small_xml_chunks(chunks, threshold=100)
            for chunk in chunks:
                cleaned_chunk = chunk.strip()
                # Annahme: Die erste Zeile entspricht dem Kapitel-Titel
                lines = [line.strip() for line in cleaned_chunk.split("\n") if line.strip()]
                chapter = lines[0] if lines else "Unbekanntes Kapitel"
                documents.append(Document(
                    page_content=cleaned_chunk,
                    metadata={"source": filename, "chapter": chapter}
                ))
    return documents


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


data_directory = r"C:\Users\fabio.cappellaro\Documents\Masterarbeit Projekt\Masterarbeit_FC\Datenpool"

# XML-Dokumente verarbeiten
xml_documents = process_xmls(data_directory)

# PDF-Dokumente verarbeiten
pdf_documents = process_pdfs(data_directory)

# Nur die PDF-Dokumente mit dem RecursiveCharacterTextSplitter in kleinere Chunks aufteilen
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=20,
    length_function=len
)
pdf_documents = text_splitter.split_documents(pdf_documents)

# Alle Dokumente kombinieren
documents = xml_documents + pdf_documents

# Embedding-Modell laden und in Vektorspeicher speichern
embedding_model = "intfloat/multilingual-e5-large"
embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
vectorstore = FAISS.from_documents(documents, embeddings)

print(f"{len(documents)} Dokument-Chunks wurden verarbeitet und in den Vektorspeicher geladen.")


# Debugging: Exportiere Daten für zwischen Kontrolle, speichert in den Ordner ExportetData
def export_data(documents, vectorstore, directory):
    os.makedirs(directory, exist_ok=True)

    # Parsed exportieren
    parsed_text_path = os.path.join(directory, "parsed_text.txt")
    with open(parsed_text_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.page_content + "\n\n")

    # Dokument-Chunks exportieren
    chunks_path = os.path.join(directory, "chunks.txt")
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write("Anzahl Chunks: " + str(len(documents)) + "\n\n")
        for doc in documents:
            f.write(f"[{doc.page_content}]\n\n")

    # Embeddings exportieren
    embeddings_path = os.path.join(directory, "embeddings.txt")
    with open(embeddings_path, "w", encoding="utf-8") as f:
        f.write("Anzahl Embeddings: " + str(vectorstore.index.ntotal) + "\n\n")
        for doc_id, embedding in enumerate(vectorstore.index.reconstruct_n(0, len(documents))):
            f.write(f"Document {doc_id}: {embedding}\n")

    print(f"Export abgeschlossen. Dateien gespeichert in: {directory}")

export_directory = r"C:\Users\fabio.cappellaro\Documents\Masterarbeit Projekt\Masterarbeit_FC\ExportedData"
export_data(documents, vectorstore, export_directory)