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
    # 1) Frage vom User
    query = input("\nBitte geben Sie Ihre Frage ein: ")
    
    # 2) Chain ausführen
    result = qa_chain.invoke(query)

    # 3) Ergebnis-Parsing
    answer = result.get('result', "Keine Antwort gefunden.")
    source_docs = result.get('source_documents', [])

    # 4) Ausgabe formatieren
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
            print(f"{i}. Quelle: {source_info} | Seite: {page_info}")
    print("-"*50)




# Debugging: Query und Result in einer Datei speichern
    try:
        with open("debug_data.json", "w", encoding="utf-8") as f:
            json.dump({"query": query, "answer": answer}, f, ensure_ascii=False, indent=2)
        print("\nDebug-Daten gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der Debug-Daten: {e}")




if __name__ == "__main__":
    # Aufruf der Funktion
    frage_und_antwort()
