import pytest
from my_app.ingestion.pdf_ingestion import get_pdf_documents

def test_ingestion_returns_documents():
    # Act
    docs = get_pdf_documents()
    
    # Assert
    # We check if it's a list. If you have files in src/my_app/ingestion/pdf_files/, 
    # len(docs) should be > 0.
    assert isinstance(docs, list)
    print(f"Captured {len(docs)} documents during test.")

def test_document_structure():
    docs = get_pdf_documents()
    if len(docs) > 0:
        # Check that LangChain objects have the expected attributes
        assert hasattr(docs[0], 'page_content')
        assert hasattr(docs[0], 'metadata')