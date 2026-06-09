import sys
import traceback

print("=== Python Info ===")
print(sys.version)

try:
    print("\n=== Checking imports ===")
    import streamlit
    print(f"streamlit: {streamlit.__version__}")
    import langchain
    print(f"langchain: {langchain.__version__}")
    import chromadb
    print(f"chromadb: {chromadb.__version__}")
    import PyPDF2
    print(f"PyPDF2: {PyPDF2.__version__}")
    import docx
    print(f"python-docx: OK")
    import ollama
    print(f"ollama: OK")
    print("All imports OK")
except Exception as e:
    print(f"Import error: {e}")
    traceback.print_exc()

try:
    print("\n=== Testing Ollama connection ===")
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=10)
    print(f"Ollama response: {r.status_code}")
    data = r.json()
    print(f"Models: {[m['name'] for m in data.get('models', [])]}")
except Exception as e:
    print(f"Ollama error: {e}")
    traceback.print_exc()

try:
    print("\n=== Testing OllamaEmbeddings ===")
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    result = embeddings.embed_query("test hello")
    print(f"Embedding dim: {len(result)}")
except Exception as e:
    print(f"Embedding error: {e}")
    traceback.print_exc()

try:
    print("\n=== Testing Chroma ===")
    import tempfile, os, shutil
    tmpdir = tempfile.mkdtemp()
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_texts(
        texts=["Hello world", "This is a test"],
        embedding=embeddings,
        persist_directory=tmpdir
    )
    print("Chroma created")
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke("hello")
    print(f"Retriever found: {len(docs)} docs")
    try:
        vectordb.persist()
        print("Persist OK")
    except Exception as pe:
        print(f"Persist warning (may not be needed): {pe}")
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("Chroma test passed")
except Exception as e:
    print(f"Chroma error: {e}")
    traceback.print_exc()

try:
    print("\n=== Testing OllamaLLM ===")
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model="qwen2:0.5b", temperature=0.1)
    result = llm.invoke("Say hello in Chinese")
    print(f"LLM result: {result[:200]}")
except Exception as e:
    print(f"LLM error: {e}")
    traceback.print_exc()

print("\n=== DONE ===")
