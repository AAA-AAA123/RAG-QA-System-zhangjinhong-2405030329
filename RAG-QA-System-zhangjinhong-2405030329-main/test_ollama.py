import sys
from langchain_ollama import OllamaLLM

def test_ollama_connection():
    try:
        llm = OllamaLLM(model="deepseek-r1:7b")
        response = llm.invoke("Hello! How are you?")
        print("Ollama连接成功!")
        print("响应结果:", response)
        return True
    except Exception as e:
        print(f"Ollama连接失败: {e}")
        print("\n请确保：")
        print("1. Ollama已安装")
        print("2. Ollama服务已启动")
        print("3. deepseek-r1:7b模型已下载（执行: ollama pull deepseek-r1:7b）")
        print("4. nomic-embed-text嵌入模型已下载（执行: ollama pull nomic-embed-text）")
        return False

def test_embedding():
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        text = "测试嵌入模型"
        result = embeddings.embed_query(text)
        print(f"嵌入模型测试成功! 向量维度: {len(result)}")
        return True
    except Exception as e:
        print(f"嵌入模型测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== Ollama连接测试 ===")
    print("\n1. 测试LLM连接...")
    llm_ok = test_ollama_connection()
    
    print("\n2. 测试嵌入模型...")
    embed_ok = test_embedding()
    
    if llm_ok and embed_ok:
        print("\n✅ 所有测试通过!")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查Ollama安装和模型下载")
        sys.exit(1)