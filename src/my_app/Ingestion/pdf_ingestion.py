from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from my_app.utils.logger import get_logger

logger = get_logger(__name__)

# 1. Define the loader
dir_loader = DirectoryLoader(
    "pdf_files",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)

# 2. Wrap the execution in a function
def get_pdf_documents():
    """This function 'exports' the documents to whoever calls it."""
    logger.info("Starting PDF directory load...")
    
    docs = dir_loader.load()
    
    logger.info(f"Loaded {len(docs)} document pages.")
    return docs

# 3. This allows you to still test THIS file directly
if __name__ == "__main__":
    docs = get_pdf_documents()
    print(docs[0].page_content[:300])