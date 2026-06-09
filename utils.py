import os
import PyPDF2
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def load_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def load_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return load_pdf(file_path)
    elif ext == '.docx':
        return load_docx(file_path)
    elif ext == '.txt':
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_documents_from_folder(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        return documents

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.pdf', '.docx', '.txt']:
                try:
                    text = load_document(file_path)
                    if text.strip():
                        documents.append({"filename": filename, "content": text})
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

def create_vector_db(documents, persist_directory="./faiss_db"):
    all_chunks = []
    for doc in documents:
        chunks = split_text(doc["content"])
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No text chunks to process")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = FAISS.from_texts(texts=all_chunks, embedding=embeddings)
    os.makedirs(persist_directory, exist_ok=True)
    vectordb.save_local(persist_directory)
    return vectordb

def get_retriever(persist_directory="./faiss_db", k=3):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectordb.as_retriever(search_kwargs={"k": k})

def create_qa_chain(retriever):
    llm = OllamaLLM(model="qwen2:0.5b", temperature=0.1)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    template = """基于提供的参考文档回答问题。
请严格按照文档内容进行回答，不要添加任何外部知识。
如果文档中没有相关信息，请明确说"文档中未找到相关答案"。

参考文档：
{context}

问题：{question}

回答："""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa_chain

def query_qa_chain(qa_chain, question):
    try:
        result = qa_chain.invoke({"question": question})
        return result.get("answer", "文档中未找到相关答案")
    except Exception as e:
        return f"查询失败: {e}"
