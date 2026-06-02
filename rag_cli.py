import os
import sys
import shutil
from utils import load_documents_from_folder, create_vector_db, get_retriever, create_qa_chain, query_qa_chain

def create_sample_documents(docs_folder):
    sample_docs = [
        {
            "filename": "nlp_intro.txt",
            "content": """自然语言处理（Natural Language Processing，简称NLP）是人工智能领域的一个重要分支，
它致力于使计算机能够理解、解释和生成人类语言。NLP技术结合了计算机科学、语言学和机器学习等多个领域的知识。

NLP的主要应用包括：
1. 机器翻译：将一种语言翻译成另一种语言
2. 情感分析：分析文本中的情感倾向
3. 命名实体识别：识别文本中的人名、地名、组织机构等实体
4. 文本分类：将文本分类到预定义的类别中
5. 问答系统：回答用户提出的问题

近年来，深度学习技术的发展极大地推动了NLP的进步，Transformer架构的出现更是革命性的突破。
BERT、GPT等预训练模型在各种NLP任务上取得了优异的成绩。"""
        },
        {
            "filename": "transformer.txt",
            "content": """Transformer是一种基于自注意力机制的深度学习模型，由Google在2017年提出。
它摒弃了传统的循环神经网络结构，采用了全新的注意力机制，能够更好地捕捉文本中的长距离依赖关系。

Transformer的主要组成部分包括：
- 编码器（Encoder）：负责处理输入序列
- 解码器（Decoder）：负责生成输出序列
- 多头注意力机制（Multi-Head Attention）：允许模型同时关注输入的不同部分
- 前馈神经网络（Feed-Forward Neural Network）：对每个位置进行独立的非线性变换

Transformer架构的优势在于其并行计算能力，相比RNN可以更高效地训练模型。
这使得训练更大规模的模型成为可能，从而推动了GPT、BERT等大型语言模型的发展。"""
        },
        {
            "filename": "bert.txt",
            "content": """BERT（Bidirectional Encoder Representations from Transformers）是由Google于2018年发布的预训练语言模型。
它基于Transformer的编码器部分，采用双向注意力机制来理解上下文。

BERT的预训练任务包括：
1. 掩码语言模型（Masked Language Modeling）：随机掩盖输入中的一些token，让模型预测被掩盖的token
2. 下一句预测（Next Sentence Prediction）：判断两个句子是否连续

BERT在多个NLP任务上取得了突破性进展，包括问答、命名实体识别、文本分类等。
它的出现标志着预训练模型时代的到来，为后续的NLP研究奠定了基础。"""
        },
        {
            "filename": "gpt.txt",
            "content": """GPT（Generative Pre-trained Transformer）是由OpenAI开发的一系列大型语言模型。
GPT模型基于Transformer的解码器架构，采用自回归的方式生成文本。

GPT系列模型的发展历程：
- GPT-1：1.17亿参数，展示了预训练模型的潜力
- GPT-2：15亿参数，能够生成连贯的长文本
- GPT-3：1750亿参数，在多种任务上达到了接近人类的水平
- GPT-4：进一步提升了多模态理解能力

GPT模型采用"预训练+微调"的范式，在海量文本数据上进行预训练，
然后通过少量标注数据进行微调即可适应各种下游任务。"""
        },
        {
            "filename": "sentiment_analysis.txt",
            "content": """情感分析（Sentiment Analysis）是NLP的一个重要应用领域，
它旨在自动识别和提取文本中的情感信息，判断文本表达的是正面、负面还是中性的情感。

情感分析的主要方法包括：
1. 基于词典的方法：使用情感词典来计算文本的情感分数
2. 机器学习方法：使用分类算法如SVM、朴素贝叶斯等进行训练
3. 深度学习方法：使用CNN、RNN、Transformer等模型进行端到端训练

情感分析的应用场景包括：
- 社交媒体监控：分析用户对产品或事件的看法
- 客户反馈分析：了解客户对产品的满意度
- 舆情分析：监测公众对特定话题的情绪倾向

近年来，预训练语言模型在情感分析任务上表现出色，
通过微调BERT、GPT等模型可以获得很高的准确率。"""
        }
    ]
    
    for doc in sample_docs:
        with open(os.path.join(docs_folder, doc["filename"]), 'w', encoding='utf-8') as f:
            f.write(doc["content"])
    print("已创建示例文档")

def main():
    print("=== RAG智能问答系统 - 命令行版本 ===")
    
    docs_folder = "./documents"
    if not os.path.exists(docs_folder):
        print(f"文档文件夹 {docs_folder} 不存在，正在创建...")
        os.makedirs(docs_folder)
        create_sample_documents(docs_folder)
    
    print("\n正在加载文档...")
    documents = load_documents_from_folder(docs_folder)
    print(f"成功加载 {len(documents)} 份文档")
    
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    print("\n正在构建向量数据库...")
    try:
        create_vector_db(documents)
        print("向量数据库构建完成")
    except Exception as e:
        print(f"构建向量数据库失败: {e}")
        sys.exit(1)
    
    print("\n正在创建问答链...")
    retriever = get_retriever()
    qa_chain = create_qa_chain(retriever)
    print("问答链创建完成")
    
    print("\n=== 开始问答（输入 'quit' 退出）===")
    while True:
        try:
            question = input("\n请输入问题：")
            if question.lower() == 'quit':
                print("退出问答系统")
                break
            
            print("正在思考...")
            answer = query_qa_chain(qa_chain, question)
            print(f"回答：{answer}")
        except KeyboardInterrupt:
            print("\n退出问答系统")
            break
        except Exception as e:
            print(f"问答过程出错: {e}")

if __name__ == "__main__":
    main()