# rag/loader.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return one Document per page.
    Each Document has:
        .page_content  → extracted text
        .metadata      → {'source': 'path/to/file.pdf', 'page': 0}
    Note: page is 0-indexed (page 1 of PDF = metadata['page'] == 0)
    """
    loader = PyPDFLoader(file_path)
    pages  = loader.load()
    return pages