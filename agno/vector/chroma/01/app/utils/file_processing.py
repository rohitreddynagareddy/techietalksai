from typing import List
from pypdf import PdfReader
from docx import Document
from langchain.schema import Document as LangDocument
from langchain.text_splitter import TextSplitter

def process_files(pdf_files, docx_files, text_splitter: TextSplitter) -> List[LangDocument]:
    documents = []
    
    # Process PDF files
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        docs = text_splitter.create_documents([text])
        documents.extend(docs)
    
    # Process DOCX files
    for docx_file in docx_files:
        doc = Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        docs = text_splitter.create_documents([text])
        documents.extend(docs)
    
    return documents
