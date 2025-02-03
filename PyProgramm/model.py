from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

# Multi-Query-Retriever kommt aus retriever.py
from retriever import get_multi_query_retriever

# PromptTemplate für deine finale Antwort
from langchain.prompts import PromptTemplate

# Mistral LLM
login("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")

model_name = "mistralai/Ministral-8B-Instruct-2410"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, legacy=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    token=True
)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=600,
    do_sample=True,
    temperature=0.6,
    return_full_text=False,
    top_k=50,
    top_p=0.9,
    #repetition_penalty= ,
    #batch_size=4,
    # eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=text_pipeline)

# Propt-Template zum Generieren der finalen Antwort
template_de_kurz = """Du bist ein hilfreicher KI-Assistent für technische Dokumentationen. 
Beantworte die folgende Frage ausschließlich basierend auf dem untenstehenden Kontext. 
- Falls der Kontext mehrere relevante Informationen enthält, fasse sie präzise zusammen. 
- Falls der Kontext keine direkte Antwort enthält, gib nur "Keine Antwort." zurück.
- Füge keine zusätzlichen Informationen hinzu, die nicht im Kontext vorhanden sind.

Frage:
{question}

Kontext:
{context}

Antwort:
"""
prompt = PromptTemplate(template=template_de_kurz, input_variables=["context", "question"])

#Retriver und Chain
multi_query_retriever = get_multi_query_retriever(llm)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=multi_query_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=False,  
)