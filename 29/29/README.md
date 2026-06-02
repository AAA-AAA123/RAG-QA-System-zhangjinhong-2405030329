# RAG智能问答系统

基于本地知识库的检索增强生成（RAG）智能问答系统，支持使用Ollama本地大模型进行文档问答。

## 功能特点

- 📄 支持PDF、DOCX和TXT文档上传
- 🔍 基于Chroma向量数据库的文档检索
- 🤖 集成Ollama本地大模型（支持deepseek-r1:7b、qwen2:7b等）
- 💬 支持多轮对话记忆
- 📊 实时显示知识库状态
- 🎨 友好的Web界面（基于Streamlit）

## 环境要求

- Python 3.8+
- Ollama（已安装并下载模型）

## 安装步骤

1. **安装Ollama**
   - 下载地址：https://ollama.com/download
   - 安装完成后，下载模型：
     ```bash
     ollama pull deepseek-r1:7b
     ollama pull nomic-embed-text
     ```

2. **克隆项目**
   ```bash
   git clone https://github.com/yourusername/RAG-QA-System.git
   cd RAG-QA-System
   ```

3. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

4. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

### 运行Web应用
```bash
streamlit run app.py
```

### 使用步骤
1. 打开浏览器访问显示的URL（通常是 http://localhost:8501）
2. 点击"浏览文件"上传PDF、DOCX或TXT文档
3. 点击"构建知识库"按钮处理文档
4. 在问答交互区输入问题并点击"提问"
5. 查看回答结果和对话历史

### 命令行版本
```bash
python rag_cli.py
```

### 测试Ollama连接
```bash
python test_ollama.py
```

## 关键技术点

### RAG流程
1. **文档加载**：支持PDF、DOCX和TXT格式文档的解析
2. **文本分块**：使用RecursiveCharacterTextSplitter，chunk_size=1000，chunk_overlap=200
3. **向量化**：使用Ollama的nomic-embed-text嵌入模型
4. **向量存储**：使用Chroma向量数据库
5. **检索**：基于相似度检索最相关的3个文本块
6. **生成**：使用Ollama大模型基于检索结果生成回答

### 模型配置
- 嵌入模型：nomic-embed-text（Ollama内置）
- 语言模型：deepseek-r1:7b（可替换为qwen2:7b等其他模型）
- 向量数据库：Chroma

## 项目结构

```
RAG-QA-System/
├── app.py              # Streamlit Web应用主文件
├── utils.py            # 工具函数（文档加载、向量化、问答链等）
├── rag_cli.py          # 命令行版本问答脚本
├── test_ollama.py      # Ollama连接测试脚本
├── requirements.txt    # 依赖列表
├── setup.py            # PyInstaller打包配置
├── .gitignore          # Git忽略配置
└── documents/          # 示例文档目录（运行rag_cli.py时自动创建）
```

## 测试示例

### 相关问题测试
1. **问**：什么是自然语言处理？
   **答**：自然语言处理（Natural Language Processing，简称NLP）是人工智能领域的一个重要分支，它致力于使计算机能够理解、解释和生成人类语言。

2. **问**：Transformer模型的主要组成部分有哪些？
   **答**：Transformer的主要组成部分包括编码器、解码器、多头注意力机制和前馈神经网络。

3. **问**：BERT的预训练任务是什么？
   **答**：BERT的预训练任务包括掩码语言模型和下一句预测。

4. **问**：GPT模型采用什么架构？
   **答**：GPT模型基于Transformer的解码器架构，采用自回归的方式生成文本。

5. **问**：情感分析有哪些应用场景？
   **答**：情感分析的应用场景包括社交媒体监控、客户反馈分析和舆情分析。

### 无关问题测试
1. **问**：今天天气怎么样？
   **答**：文档中未找到相关答案

2. **问**：如何制作蛋糕？
   **答**：文档中未找到相关答案

## 打包成EXE

使用PyInstaller将应用打包成独立的可执行文件：

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name RAG-QA-System app.py
```

打包完成后，可执行文件位于`dist/RAG-QA-System.exe`。

## 已知问题与改进方向

- 当前仅支持PDF、DOCX和TXT格式，可扩展支持更多格式
- 可添加夜间模式功能
- 可添加批量上传和问答记录导出功能
- 可优化向量数据库的检索效率

## 许可证

MIT License

## 作者

姓名：[你的姓名]
学号：[你的学号]

---

**注意**：运行本系统前请确保Ollama服务已启动，并且所需模型已下载完成。