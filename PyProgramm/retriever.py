import logging
from typing import List
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import BaseOutputParser

from data import vectorstore


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser für eine Liste von Zeilen."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Entfernt leere Zeilen


logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)

# Prompt-Vorlage, um mehrere Varianten einer Frage zu generieren
multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Du bist ein KI-Assistent für technische Dokumentationen. Generiere drei alternative Formulierungen der Frage "{question}" zur Suche relevanter Dokumente in einer Vektordatenbank. Verwende dabei alternative Fachbegriffe und Bezeichnungen, ohne die ursprüngliche Bedeutung zu verändern.
"""
)

def get_multi_query_retriever(llm):
    # Normaler Retriever
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3, 'score_threshold': 0.8}
    )

    # Output-Parser
    output_parser = LineListOutputParser()

    # LLM-Kette mit Output-Parser
    llm_chain = multi_query_prompt | llm | output_parser

    # Multi-Query-Retriever
    multi_query_retriever = MultiQueryRetriever(
        retriever=base_retriever,
        llm_chain=llm_chain
    )

    return multi_query_retriever
