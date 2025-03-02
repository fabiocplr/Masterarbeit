import json
import torch
from data import embeddings 
from scipy.spatial.distance import cosine
from model import text_pipeline, tokenizer
from rouge_score import rouge_scorer

def cosine_similarity(vector1, vector2):
    """ Berechnet Cosine Similarity"""
    cos_sim = 1 - cosine(vector1, vector2)
    return cos_sim

def model_confidence(answer):
    """ Berechnet den Confidence Score der Antwort basierend auf Token-Wahrscheinlichkeiten. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = tokenizer(answer, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = text_pipeline.model(**inputs)

    # Softmax auf Logits anwenden, um Wahrscheinlichkeiten zu erhalten
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Berechnung des Confidence Scores (Durchschnitt der Top-1 Token-Wahrscheinlichkeiten)
    top_probs, _ = torch.max(probs, dim=-1)  
    confidence_score = top_probs.mean().item()
    
    return confidence_score

#ROUGE Metrik
def rouge_score(answer, correct_answer):
    """ Berechnet ROUGE-1 und ROUGE-L"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(answer, correct_answer)
    rouge1_precision = scores["rouge1"].precision
    rouge1_recall = scores["rouge1"].recall
    rouge1_f1 = scores["rouge1"].fmeasure

    rougeL_precision = scores["rougeL"].precision
    rougeL_recall = scores["rougeL"].recall
    rougeL_f1 = scores["rougeL"].fmeasure

    return rouge1_precision, rouge1_recall, rouge1_f1, rougeL_precision, rougeL_recall, rougeL_f1  
    


def debug_info(query, result_dict, correct_answer):
    """ Zeigt Debug-Infos an: Kontext, Cosine Similarity, Confidence Score und ROUGE Score. """
    
    # Kontext ausgeben
    source_docs = result_dict.get("source_documents", [])
    if not source_docs:
        print("Keine Dokumente in den Ergebnissen vorhanden!")
        return

    query_vector = embeddings.embed_query(query)
    
    similarity_results = []
    for doc in source_docs:
        doc_text = doc.get("text", "")
        doc_vector = embeddings.embed_documents([doc_text])[0]

        cos_sim = cosine_similarity(query_vector, doc_vector)

        similarity_results.append({
            "metadata": {
                "source": doc.get("source", "Unbekannte Quelle"),
                "chapter": doc.get("chapter", "Unbekanntes Kapitel"),
                "page": doc.get("page", "N/A")
            },
            "text": doc_text,
            "cosine_similarity": cos_sim
        })

    # Chunks nach Cosine Similarity absteigend sortieren
    similarity_results = sorted(similarity_results, key=lambda x: x["cosine_similarity"], reverse=True)

    # Vollständige Ausgabe des relevantesten Kontextblocks
    print("\n=== Vollständiger Kontext (sortiert nach Relevanz) ===")
    for i, doc in enumerate(similarity_results, start=1):
        metadata = doc["metadata"]
        print(f"\n--- Kontextblock {i} ---")
        print(f"Quelle: {metadata['source']}, Kapitel: {metadata['chapter']}, Seite: {metadata['page']}")
        print(f"Cosine Similarity: {doc['cosine_similarity']:.4f}")
        print(f"\n{doc['text']}\n")
        print("-" * 50)

    # Confidence Score berechnen
    print("\n=== Confidence Score ===")
    answer = result_dict.get("result", "Keine Antwort gefunden.")
    confidence_score = model_confidence(answer)
    print(f"Confidence Score der Antwort: {confidence_score:.4f}")


    # ROUGE-1 & ROUGE-L Score berechnen
    rouge1_precision, rouge1_recall, rouge1_f1, rougeL_precision, rougeL_recall, rougeL_f1 = rouge_score(answer, correct_answer)


    print("\n=== ROUGE Scores ===")
    print(f"ROUGE-1 Precision: {rouge1_precision:.4f}")
    print(f"ROUGE-1 Recall: {rouge1_recall:.4f}")
    print(f"ROUGE-1 F1-Score: {rouge1_f1:.4f}")

    print(f"\nROUGE-L Precision: {rougeL_precision:.4f}")
    print(f"ROUGE-L Recall: {rougeL_recall:.4f}")
    print(f"ROUGE-L F1-Score: {rougeL_f1:.4f}")


    # Ergebnisse in Debug-Datei speichern
    debug_output = {
        "query": query,
        "answer": answer,
        "confidence_score": confidence_score,
        "rouge1_precision": rouge1_precision,
        "rouge1_recall": rouge1_recall,
        "rouge1_f1": rouge1_f1,
        "rougeL_precision": rougeL_precision,
        "rougeL_recall": rougeL_recall,
        "rougeL_f1": rougeL_f1,
        "sorted_source_documents": similarity_results
    }

    with open("debug_output.json", "w", encoding="utf-8") as f:
        json.dump(debug_output, f, ensure_ascii=False, indent=2)

    print("\nDebug-Daten gespeichert in debug_output.json.")

if __name__ == "__main__":
    try:
        with open("debug_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Fehler: Keine debug_data.json gefunden! Bitte zuerst main.py ausführen.")
        exit(1)
    except json.JSONDecodeError:
        print("Fehler: debug_data.json enthält ungültige JSON-Daten!")
        exit(1)

    # Korrekte Antwort aus correct_answer.json laden
    try:
        with open("correct_answer.json", "r", encoding="utf-8") as f:
            correct_answer_data = json.load(f)
    except FileNotFoundError:
        print("Fehler: Keine correct_answer.json gefunden! Bitte die Datei erstellen.")
        correct_answer_data = {}
    except json.JSONDecodeError:
        print("Fehler: correct_answer.json enthält ungültige JSON-Daten!")
        correct_answer_data = {}
    

    query = data.get("query", "Keine Frage gefunden.")
    result_dict = {
        "result": data.get("answer", "Keine Antwort gefunden."),
        "source_documents": data.get("Kontext", [])
    }

    # Hole die korrekte Antwort aus der JSON-Datei
    correct_answer = correct_answer_data.get(query, "Keine Referenzantwort vorhanden.")

    print(f"\n=== DEBUGGING-SKRIPT ===")
    print(f"Query aus main.py war: '{query}'")

    debug_info(query, result_dict, correct_answer)
