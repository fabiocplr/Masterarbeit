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
import os
import xml.etree.ElementTree as ET

namespaces = {
    "n": "http://www.schema.de/2004/ST4/XmlImportExport/Node",
    "d": "http://www.schema.de/2004/ST4/XmlImportExport/Data",
    "l": "http://www.schema.de/2004/ST4/XmlImportExport/Link",
    "m": "http://www.schema.de/2004/ST4/XmlImportExport/Meta"
}

# Definiere die Container-Tags, die als Abschnitte gelten sollen.
container_tags = {
    "TextFolder", "Project", "Document",
    "InfoType01", "InfoType02", "InfoType03",
    "InfoType04", "InfoType05", "InfoType07",
    "Content"
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

def tag_without_ns(elem):
    if elem.tag.startswith("{"):
        return elem.tag.split("}", 1)[1]
    return elem.tag

def extract_sections_with_parent_map(root):
    """
    Sucht in der gesamten XML nach Container-Elementen (definiert in container_tags),
    die als inhaltlich zusammengehörige Abschnitte gelten, und extrahiert in der Reihenfolge
    (basierend auf dem l:Sort-Attribut) alle untergeordneten Data-Title und Data-Content-Einträge
    (nur Aspect="de"). Dabei werden Überschriften (Data-Title) und zugehörige Inhalte gruppiert.
    """
    # Erstelle eine Parent-Map, da ElementTree kein getparent() bietet
    parent_map = {child: parent for parent in root.iter() for child in list(parent)}
    
    def has_container_ancestor(elem):
        current = parent_map.get(elem)
        while current:
            if tag_without_ns(current) in container_tags:
                return True
            current = parent_map.get(current)
        return False

    sections = []
    # Sammle alle Container, die keine untergeordneten Container sind
    for elem in root.iter():
        if tag_without_ns(elem) in container_tags and not has_container_ancestor(elem):
            # Lese den Sortierwert (l:Sort) des Containers
            sort_val = elem.attrib.get("{http://www.schema.de/2004/ST4/XmlImportExport/Link}Sort")
            try:
                sort_val = float(sort_val) if sort_val is not None else 0
            except ValueError:
                sort_val = 0

            section_parts = []
            # Falls vorhanden: Container-Titel aus dem Attribut "Title"
            title_attr = elem.attrib.get("{http://www.schema.de/2004/ST4/XmlImportExport/Node}Title")
            if title_attr:
                section_parts.append(title_attr.strip())

            # Innerhalb des Containers: Sammle alle untergeordneten Data-Title und Data-Content-Elemente,
            # allerdings nur jene, die den Aspekt "de" betreffen.
            # Wir bauen dabei Gruppen: Jede Data-Title markiert eine Überschrift, der nachfolgende Text
            # (aus Data-Content und ggf. weiteren Data-Title-Einträgen) wird gruppiert.
            groups = []
            current_heading = None
            current_chunk = []
            # Iteriere über alle Nachfahren in Dokumentreihenfolge
            for sub in elem.iter():
                sub_tag = tag_without_ns(sub)
                if sub_tag in {"Data-Title", "Data-Content"}:
                    # Filtere nach Value-Elementen mit n:Aspect="de"
                    for value in sub.findall("n:Value[@n:Aspect='de']", namespaces):
                        # Gehe alle darin enthaltenen Entry-Elemente durch
                        for entry in value.findall("n:Entry", namespaces):
                            text = extract_content(entry)
                            if not text:
                                continue
                            if sub_tag == "Data-Title":
                                # Wenn bereits eine Überschrift vorliegt, sichere die bisherige Gruppe
                                if current_heading or current_chunk:
                                    groups.append((current_heading, "\n\n".join(current_chunk)))
                                    current_chunk = []
                                current_heading = text
                            else:  # Data-Content
                                current_chunk.append(text)
            # Sichere die letzte Gruppe, falls vorhanden
            if current_heading or current_chunk:
                groups.append((current_heading, "\n\n".join(current_chunk)))

            # Führe die Gruppen zusammen: Überschrift (falls vorhanden) gefolgt von zugehörigen Inhalten
            group_texts = []
            for heading, content in groups:
                if heading and content:
                    group_texts.append(heading + "\n\n" + content)
                elif heading:
                    group_texts.append(heading)
                else:
                    group_texts.append(content)
            if group_texts:
                section_parts.append("\n\n".join(group_texts))

            # Verbinde alle Teile des Containers mit doppelten Zeilenumbrüchen
            if section_parts:
                full_text = "\n\n".join(section_parts)
                sections.append((sort_val, full_text))

    # Sortiere die Container entsprechend dem l:Sort-Wert
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
            # Hier wird angenommen, dass du eine Document-Klasse hast (z.B. aus langchain)
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
