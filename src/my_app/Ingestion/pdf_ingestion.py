import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from my_app.utils.logger import get_logger

logger = get_logger(__name__)

# Locate the PDF folder relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "pdf_files")

dir_loader = DirectoryLoader(
    pdf_path,
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)

def get_pdf_documents():
    logger.info(f"Loading PDFs from: {pdf_path}")
    documents = dir_loader.load()
    logger.info(f"Loaded {len(documents)} document pages.")
    return documents

if __name__ == "__main__":
    docs = get_pdf_documents()
    if docs:
        print(docs[0].page_content[:300])