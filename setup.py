import sys
from cx_Freeze import setup, Executable

base = None
if sys.platform == "win32":
    base = "Win32GUI"

executables = [Executable("app.py", base=base, target_name="RAG-QA-System.exe")]

setup(
    name="RAG-QA-System",
    version="1.0",
    description="基于本地知识库的RAG智能问答系统",
    executables=executables,
    options={
        "build_exe": {
            "packages": ["streamlit", "langchain", "chromadb", "PyPDF2", "docx", "ollama"],
            "include_files": [],
            "excludes": []
        }
    }
)