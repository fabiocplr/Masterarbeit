import logging
from typing import List
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import BaseOutputParser

from data import vectorstore


logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# class LineListOutputParser(BaseOutputParser[List[str]]):
#     """Output parser for a list of lines."""

#     def parse(self, text: str) -> List[str]:
#         lines = text.strip().split("\n")
#         return list(filter(None, lines))  # Remove empty lines
    

# Prompt-Vorlage, um mehrere Varianten einer Frage zu generieren
multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Du bist ein KI-Assistent, spezialisiert auf technische Dokumentationen.  
Deine Aufgabe ist es, **drei alternative Versionen** der folgenden technischen Frage zu generieren: "{question}"  

- Verwende **technisch präzise Formulierungen** oder wandle die Frage leicht um.  
- Behalte den **ursprünglichen Sinn** bei und füge **keine neuen Informationen** hinzu.  
- Achte darauf, dass die Fragen für die Nutzung in **technischen Handbüchern, Anleitungen oder Wartungsdokumenten** geeignet sind.  
- Falls relevante Begriffe vorhanden sind, verwende ggf. **fachspezifische Synonyme**.   
""".strip()
)

def get_multi_query_retriever(llm):
    """
    Erstellt (und gibt) einen Multi-Query-Retriever zurück, 
    der den globalen Vectorstore aus Datenspeicher.py verwendet.
    """
    # Normaler Retriever
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5, 'score_threshold': 0.8}
    )

    # Multi-Query-Retriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=multi_query_prompt

    )

    return multi_query_retriever

