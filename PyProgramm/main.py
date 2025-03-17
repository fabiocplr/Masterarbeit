from model import qa_chain
import json
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API")
os.environ["LANGCHAIN_PROJECT"] = "Master"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


def frage_und_antwort():
    # Frage vom User
    query = input("\nBitte geben Sie Ihre Frage ein: ")
    
    # Chain ausführen
    result = qa_chain.invoke(query)

    # Ergebnis-Parsing
    answer = result.get('result', "Keine Antwort gefunden.")
    source_docs = result.get('source_documents', [])

    # Ausgabe formatieren
    print("\n" + "="*50)
    print("Antwort auf deine Frage:")
    print("="*50)
    print(answer)

    if source_docs:
        print("\n" + "-"*50)
        print("Quellen (Auszug):")
        print("-"*50)
        for i, doc in enumerate(source_docs, start=1):
            metadata = doc.metadata
            source_info = metadata.get("source", "Unbekannte Quelle")
            page_info = metadata.get("page", "N/A")
            chapter_info = metadata.get("chapter", "Unbekanntes Kapitel")
            print(f"{i}. Quelle: {source_info} | Kapitel: {chapter_info} | Seite: {page_info}")
    print("-"*50)




# Evaluation: Query und Result in einer Datei speichern
    try:
        with open("evaluation_data.json", "w", encoding="utf-8") as f:
            json.dump({
                        "query": query, 
                        "answer": answer, 
                        "Kontext": [
                            {
                                "text": doc.page_content,  
                                "source": doc.metadata.get("source", "Unbekannte Quelle"),
                                "page": doc.metadata.get("page", "N/A"),
                                "chapter": doc.metadata.get("chapter", "Unbekanntes Kapitel"),
                                "id": doc.metadata.get("id", "Unbekannte ID")
                            }
                            for doc in source_docs
                        ]
                    }, f, ensure_ascii=False, indent=2)

        print("\nEvaluation-Daten gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der Evaluation-Daten: {e}")




if __name__ == "__main__":
    # Aufruf der Funktion
    frage_und_antwort()
