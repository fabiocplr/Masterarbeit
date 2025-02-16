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

# XML Parsen
# Namespace-Deklarationen extrahieren

namespaces = {
    "n": "http://www.schema.de/2004/ST4/XmlImportExport/Node",
    "d": "http://www.schema.de/2004/ST4/XmlImportExport/Data",
    "l": "http://www.schema.de/2004/ST4/XmlImportExport/Link",
    "m": "http://www.schema.de/2004/ST4/XmlImportExport/Meta"
}

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

def remove_redundant_blank_lines(file_content):
    """Entfernt überflüssige Leerzeilen (max. 2 aufeinanderfolgende)."""
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

def extract_sections_with_parent_map(root):
    """
    Sucht in der gesamten XML nach Container-Elementen, die als inhaltlich zusammengehörige Abschnitte gelten.
    Hierzu zählen beispielsweise Elemente mit den Tags TextFolder, Project, Document, InfoType01, InfoType04 oder Content.
    
    Für jeden solchen Container wird (falls vorhanden) zunächst der Title (Attribut "Title") eingefügt und
    anschließend alle untergeordneten Data-Title und Data-Content (Aspect="de") in der Reihenfolge ihres Auftretens zusammengeführt.
    Zwischen den einzelnen Knoten wird ein Absatz (doppelte Zeilenumbrüche) eingefügt.
    """
    container_tags = {"TextFolder", "Project", "Document", "InfoType01", "InfoType04", "Content"}
    
    def tag_without_ns(elem):
        if elem.tag.startswith("{"):
            return elem.tag.split("}", 1)[1]
        return elem.tag
    
    # Erstellen einer Parent-Map, da ET keine getparent()-Methode bietet
    parent_map = {child: parent for parent in root.iter() for child in list(parent)}
    
    def has_container_ancestor(elem):
        current = parent_map.get(elem)
        while current:
            if tag_without_ns(current) in container_tags:
                return True
            current = parent_map.get(current)
        return False
    
    sections = []
    # Mit enumerate erfassen wir die Position im Dokument
    for i, elem in enumerate(root.iter()):
        if tag_without_ns(elem) in container_tags and not has_container_ancestor(elem):
            section_parts = []
            # Zuerst: Falls vorhanden, den Titel des Containers (Attribut "Title")
            title_attr = elem.attrib.get("{http://www.schema.de/2004/ST4/XmlImportExport/Node}Title")
            if title_attr:
                section_parts.append(title_attr.strip())
            # Dann: Alle untergeordneten Data-Title und Data-Content-Elemente (Aspect="de") in Dokumentreihenfolge
            for sub in elem.iter():
                if tag_without_ns(sub) in {"Data-Title", "Data-Content"}:
                    for value in sub.findall("n:Value[@n:Aspect='de']", namespaces):
                        for entry in value.findall("n:Entry", namespaces):
                            text = extract_content(entry)
                            if text:
                                section_parts.append(text.strip())
            if section_parts:
                # Hier werden die einzelnen Teile mit doppeltem Zeilenumbruch (als Absatz-Trenner) verbunden
                section_text = "\n\n".join(section_parts)
                sections.append((i, section_text))
    # Sortiere die Abschnitte nach ihrer Position im Dokument
    sections.sort(key=lambda x: x[0])
    return [text for _, text in sections]

def process_xmls(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            try:
                tree = ET.parse(file_path)
            except ET.ParseError as e:
                print(f"Fehler beim Parsen von {filename}: {e}")
                continue
            root = tree.getroot()
            sections = extract_sections_with_parent_map(root)
            combined_content = "\n\n".join(sections)
            cleaned_content = remove_redundant_blank_lines(combined_content)
            documents.append(Document(page_content=cleaned_content, metadata={"source": filename}))
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
export_directory = r"C:\Users\fabio.cappellaro\Documents\Masterarbeit Projekt\Masterarbeit_FC\ExportedData"
export_data(docs, vectorstore, export_directory)
