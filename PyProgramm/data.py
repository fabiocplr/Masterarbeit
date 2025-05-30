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

# Rekursive Extraktion des Textinhalts eines Elements
def extract_content(element):
    content = []
    if element.text and element.text.strip():
        content.append(element.text.strip())
    for child in element:
        content.append(extract_content(child))
    if element.tail and element.tail.strip():
        content.append(element.tail.strip())
    return " ".join(content).strip()

# Extrahiert deutschsprachige Inhalte aus XML-Dokumenten.
def extract_deutsch_chunks(root):
    chunks = []
    current_chunk = ""

    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]  # Namespace entfernen

        # Verarbeitung von Data-Title
        if tag == "Data-Title":
            # Falls vorher ein Chunk existierte, diesen speichern
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Neuen Chunk mit Titel beginnen
            for value in elem.findall("n:Value[@n:Aspect='de']", namespaces):
                texts = [extract_content(entry) for entry in value.findall("n:Entry", namespaces) if extract_content(entry)]
                if texts:
                    current_chunk = "\n\n".join(texts) + "\n\n"

        # Verarbeitung von Data-Content
        elif tag == "Data-Content":
            for value in elem.findall("n:Value[@n:Aspect='de']", namespaces):
                texts = [extract_content(entry) for entry in value.findall("n:Entry", namespaces) if extract_content(entry)]
                if texts:
                    # Falls current_chunk leer ist, neuer Eintrag
                    if not current_chunk.strip():
                        chunks.append("\n\n".join(texts).strip())
                    else:
                        current_chunk += "\n\n".join(texts) + "\n\n"

    # Letzten Chunk speichern, falls noch vorhanden
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# Neue Funktion: Kleine XML-Chunks (< 100 Zeichen) mit dem nächsten Chunk zusammenführen
def merge_small_xml_chunks(chunks, threshold=100):
    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i].strip()
        # Solange der aktuelle Chunk unter dem Schwellenwert liegt und es einen nächsten Chunk gibt, wird dieser an den aktuellen Chunk angehängt.
        while len(current) < threshold and i + 1 < len(chunks):
            i += 1
            current += " " + chunks[i].strip()
        merged.append(current)
        i += 1
    # Falls der letzte Chunk immer noch zu klein ist und es bereits einen vorherigen Chunk gibt, wird der letzte Chunk an den vorangehenden Chunk angehängt.
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
            for idx, chunk in enumerate(chunks):
                cleaned_chunk = chunk.strip()
                # Die erste Zeile wird als Metadatum für den Kapitel-Titel verwendet
                lines = [line.strip() for line in cleaned_chunk.split("\n") if line.strip()]
                chapter = lines[0] if lines else "Unbekanntes Kapitel"
                unique_id = f"{filename}_{idx}"
                documents.append(Document(
                    page_content=cleaned_chunk,
                    metadata={"source": filename, "chapter": chapter, "id": unique_id}
                ))
    return documents


# PDF-Parsing und Splitten
def process_pdfs(directory):
    documents = []
    # RecursiveCharacterTextSplitter definieren
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=20,
        length_function=len
    )
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)

            # PyMuPDFLoader
            pdf_loader = PyMuPDFLoader(file_path)
            pdf_documents = pdf_loader.load()

            # Dateinamen als Quelle
            for doc in pdf_documents:
                doc.metadata["source"] = filename
            
            # PDF-Dokumente in kleinere Chunks splitten
            split_docs = text_splitter.split_documents(pdf_documents)
            
            # Eindeutige IDs für jeden Chunk
            for idx, doc in enumerate(split_docs):
                doc.metadata["id"] = f"{filename}_{idx}"
                documents.append(doc)

    return documents

# Pfad zum Datenverzeichnis
data_directory = r"C:\Users\fabio.cappellaro\Documents\Masterarbeit Projekt\Masterarbeit_FC\Datenpool"

# XML-Dokumente verarbeiten
xml_documents = process_xmls(data_directory)

# PDF-Dokumente verarbeiten
pdf_documents = process_pdfs(data_directory)

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