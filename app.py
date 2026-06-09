import streamlit as st
import os
import tempfile
import shutil
from utils import load_document, split_text, create_vector_db, get_retriever, create_qa_chain, query_qa_chain

def main():
    st.set_page_config(page_title="RAG智能问答系统", page_icon="📚", layout="wide")

    st.title("📚 RAG智能问答系统")
    st.markdown("基于本地知识库的检索增强生成问答系统")

    if "documents" not in st.session_state:
        st.session_state.documents = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📤 文档上传")
        uploaded_files = st.file_uploader(
            "上传PDF、DOCX或TXT文件（可同时上传多个）",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                if file_ext in ['.pdf', '.docx', '.txt']:
                    file_exists = any(doc["filename"] == uploaded_file.name for doc in st.session_state.documents)
                    if not file_exists:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        try:
                            text = load_document(tmp_file_path)
                            if text.strip():
                                st.session_state.documents.append({
                                    "filename": uploaded_file.name,
                                    "content": text
                                })
                                st.success(f"✅ {uploaded_file.name}")
                            else:
                                st.warning(f"{uploaded_file.name} 内容为空，已跳过")
                        except Exception as e:
                            st.error(f"加载 {uploaded_file.name} 失败: {e}")
                        finally:
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)

        st.subheader("📊 知识库状态")
        st.info(f"📄 当前文档: **{len(st.session_state.documents)}**")

        total_chunks = 0
        for doc in st.session_state.documents:
            chunks = split_text(doc["content"])
            total_chunks += len(chunks)
        st.info(f"🧩 文本块: **{total_chunks}**")

        if st.session_state.documents:
            st.subheader("📋 已加载文档")
            for i, doc in enumerate(st.session_state.documents):
                st.markdown(f"- **{i+1}. {doc['filename']}**")

        st.markdown("---")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 构建知识库", type="primary", disabled=(len(st.session_state.documents) == 0)):
                with st.spinner("正在构建知识库（嵌入 + 索引）..."):
                    try:
                        if os.path.exists("./faiss_db"):
                            shutil.rmtree("./faiss_db")
                        if os.path.exists("./chroma_db"):
                            shutil.rmtree("./chroma_db")
                        create_vector_db(st.session_state.documents)
                        retriever = get_retriever()
                        st.session_state.qa_chain = create_qa_chain(retriever)
                        st.success("✅ 知识库构建完成！")
                    except Exception as e:
                        st.error(f"❌ 构建知识库失败: {e}")

        with col_btn2:
            if st.button("🗑️ 清空", type="secondary"):
                st.session_state.documents = []
                st.session_state.qa_chain = None
                st.session_state.chat_history = []
                if os.path.exists("./faiss_db"):
                    shutil.rmtree("./faiss_db")
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")
                st.success("✅ 已清空")

    with col2:
        st.subheader("💬 问答交互")

        if st.session_state.qa_chain is None:
            st.info("请先在左侧上传文档，然后点击 **构建知识库**")
        else:
            st.success("✅ 知识库已就绪，可以开始提问")

            with st.form("question_form"):
                question = st.text_input("请输入您的问题：", key="question_input")
                submit = st.form_submit_button("🧠 提问")

            if submit:
                if question.strip() == "":
                    st.warning("请输入问题")
                else:
                    with st.spinner("正在思考..."):
                        try:
                            answer = query_qa_chain(st.session_state.qa_chain, question)
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": answer
                            })
                        except Exception as e:
                            st.error(f"提问失败: {e}")

            if st.session_state.chat_history:
                st.subheader("📝 对话历史")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    idx = len(st.session_state.chat_history) - i
                    with st.expander(f"❓ 问题 {idx}", expanded=(i == 0)):
                        st.markdown(f"**问：** {chat['question']}")
                        st.markdown(f"**答：** {chat['answer']}")

if __name__ == "__main__":
    main()
