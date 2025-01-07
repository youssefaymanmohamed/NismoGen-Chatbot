import PyPDF2
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

def read_text(uploaded_file: UploadedFile) -> str:
    """Reads a text file and returns its text content."""
    return uploaded_file.getvalue().decode("utf-8") # Read the file as a string

def read_pdf(uploaded_file: UploadedFile) -> str:
    """Reads a PDF file and returns its text content."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if page_text:
            text += page_text
        else:
            print(f"Warning: No text extracted from page {page_num}")
    return text

def read_csv(uploaded_file: UploadedFile) -> str:
    """Reads a CSV file and returns its text content."""
    df = pd.read_csv(uploaded_file)
    return df.to_string()

def read_arxiv(uploaded_file: UploadedFile) -> str:
    """Reads an arXiv file and returns its text content."""
    return read_pdf(uploaded_file)

def get_file_extension(filename: str) -> str:
    """
    Extract the file extension from a given filename.
    """
    return filename.split(".")[-1] if "." in filename else ""

def read_txt(uploaded_file: UploadedFile) -> str:
    return read_text(uploaded_file)

def read_markdown(uploaded_file: UploadedFile) -> str:
    return read_text(uploaded_file)

def read_file(uploaded_file: UploadedFile) -> str:
    """Reads the uploaded file and returns its text content."""
    extension = get_file_extension(uploaded_file.name).lower()
    if extension == "pdf":
        return read_pdf(uploaded_file)
    elif extension == "csv":
        return read_csv(uploaded_file)
    elif extension == "arxiv":
        return read_arxiv(uploaded_file)
    elif extension == "txt":
        return read_txt(uploaded_file)
    elif extension == "md":
        return read_markdown(uploaded_file)
    else:
        return "Not supported file type."