print("Step 1: basic imports")
import sys
sys.stdout.flush()

import streamlit
print(f"streamlit OK: {streamlit.__version__}")
sys.stdout.flush()

import langchain
print(f"langchain OK: {langchain.__version__}")
sys.stdout.flush()

import langchain_community
print(f"langchain_community OK")
sys.stdout.flush()

import langchain_ollama
print(f"langchain_ollama OK")
sys.stdout.flush()

print("Step 2: chromadb import")
sys.stdout.flush()
import chromadb
print(f"chromadb OK: {chromadb.__version__}")
sys.stdout.flush()

print("Step 3: ollama connection")
sys.stdout.flush()
import requests
r = requests.get("http://localhost:11434/api/tags", timeout=10)
print(f"Ollama OK: {r.status_code}")
sys.stdout.flush()

print("Step 4: embedding test")
sys.stdout.flush()
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vec = embeddings.embed_query("test")
print(f"Embedding OK: {len(vec)} dims")
sys.stdout.flush()

print("Step 5: Chroma create")
sys.stdout.flush()
import tempfile, shutil
tmpdir = tempfile.mkdtemp()
print(f"tmp: {tmpdir}")
sys.stdout.flush()

from langchain_community.vectorstores import Chroma
db = Chroma.from_texts(
    texts=["Hello world", "This is a test document"],
    embedding=embeddings,
    persist_directory=tmpdir
)
print("Chroma created OK")
sys.stdout.flush()

print("Step 6: LLM test")
sys.stdout.flush()
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="qwen2:0.5b", temperature=0.1)
out = llm.invoke("Say hi in Chinese")
print(f"LLM OK: {out[:100]}")
sys.stdout.flush()

shutil.rmtree(tmpdir, ignore_errors=True)
print("ALL PASSED")
