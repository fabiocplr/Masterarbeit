from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
from retriever import create_multi_query_retriever
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv


load_dotenv()

# Mistral LLM HuggingFace Key
login(os.getenv("MISTRAL_LOGIN"))

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
    temperature=0.5,
    return_full_text=False,
    top_k=30,
    top_p=0.8,
    
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=text_pipeline)

# Prompt-Template zum Generieren der finalen Antwort (LLM)
template_de_kurz = """Du bist ein KI-Assistent für technische Dokumentationen. Beantworte die Frage basierend auf dem Kontext.

- Falls keine direkte Antwort im Kontext vorhanden ist, gib **"Keine Antwort vorhanden."** zurück.
- Fasse relevante Informationen zusammen.
- Ergänze keine Informationen, die nicht im Kontext stehen.
- Strukturiere die Antwort.

**Frage:**  
{question}  

**Kontext:**  
{context}

**Antwort:**"""
prompt = PromptTemplate(template=template_de_kurz, input_variables=["context", "question"])

#Retriver und Chain
multi_query_retriever = create_multi_query_retriever(llm)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=multi_query_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},  
)