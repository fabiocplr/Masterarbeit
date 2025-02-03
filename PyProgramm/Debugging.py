import json
from data import embeddings 
from scipy.spatial.distance import cosine, euclidean

def calculate_similarity_metrics(vector1, vector2):
    """Berechnet Cosine Similarity und euklidische Distanz."""
    cos_sim = 1 - cosine(vector1, vector2)
    euc_dist = euclidean(vector1, vector2)
    return cos_sim, euc_dist

def debug_info(query, result_dict):
    """
    Gibt die abgerufenen Dokumente/Quellen komplett aus
    und berechnet Cosine Similarity + euklidische Distanz
    zum Query-Vektor.
    """
    # 1) Kontext ausgeben
    source_docs = result_dict.get("source_documents", [])
    if not source_docs:
        print("Keine Dokumente in den Ergebnissen vorhanden!")
        return

    print("\n=== Vollständige Kontextblöcke ===")
    for i, doc_data in enumerate(source_docs, start=1):
        print(f"\n-- Kontextblock {i} --")
        print(f"Metadaten: {doc_data['metadata']}")
        print(doc_data['content'])
        print("-" * 30)

    # 2) Similarity
    print("\n=== Ähnlichkeitsmetrik pro Dokument ===")
    query_vector = embeddings.embed_query(query)

    for i, doc_data in enumerate(source_docs, start=1):
        doc_text = doc_data['content']
        doc_vector = embeddings.embed_documents([doc_text])[0]

        cos_sim, euc_dist = calculate_similarity_metrics(query_vector, doc_vector)

        # Hier Preview erzeugen, um keine Probleme mit f-Strings zu haben
        preview = doc_text[:200].replace("\n", " ")
        suffix = "..." if len(doc_text) > 200 else ""

        print(f"\n--- Dokument {i} ---")
        print(f"Metadaten: {doc_data['metadata']}")
        print(f"Cosine Similarity:    {cos_sim:.4f}")
        print(f"Euklidische Distanz:  {euc_dist:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    try:
        with open("debug_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Keine debug_data.json gefunden! Bitte zuerst main.py ausführen.")
        exit(1)

    query = data["query"]
    result_dict = data["result"]

    print(f"\n=== DEBUGGING-SKRIPT ===")
    print(f"Query aus main.py war: '{query}'")

    debug_info(query, result_dict)
