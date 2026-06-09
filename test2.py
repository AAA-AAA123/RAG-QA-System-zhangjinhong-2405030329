print("=== Test: FAISS + Ollama Embeddings")
import sys
sys.stdout.flush()

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

print("Imports OK")
sys.stdout.flush()

test_texts = [
    "自然语言处理（NLP）是人工智能的一个分支，致力于使计算机能够理解、解释和生成人类语言。",
    "Transformer模型于2017年由Vaswani等人在论文《Attention is All You Need》中提出。",
    "BERT是一种基于Transformer编码器架构的预训练语言模型，由Google在2018年提出。",
    "GPT系列模型基于Transformer解码器架构。",
    "情感分析是NLP的应用场景之一。",
]

print("Creating embeddings...")
sys.stdout.flush()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
print("Embeddings model loaded")
sys.stdout.flush()

print("Creating FAISS index...")
sys.stdout.flush()

db = FAISS.from_texts(texts=test_texts, embedding=embeddings)
print("FAISS created OK")
sys.stdout.flush()

print("Testing retrieval...")
sys.stdout.flush()

retriever = db.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke("什么是自然语言处理？")
print(f"Retrieved {len(docs)} documents")
for d in docs:
    pc = d.page_content[:60]
    print(f"  - {pc}...")
sys.stdout.flush()

print("Testing LLM inference...")
sys.stdout.flush()

llm = OllamaLLM(model="qwen2:0.5b", temperature=0.1)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
template = """基于提供的参考文档回答问题。
请严格按照文档内容进行回答，不要添加任何外部知识。
如果文档中没有相关信息，请明确说"文档中未找到相关答案"。

参考文档：
{context}

问题：{question}

回答："""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

result = qa.invoke({"question": "什么是自然语言处理？"})
print(f"Answer: {result.get('answer', 'N/A')}")
sys.stdout.flush()

print("\n=== ALL TESTS PASSED ===")
