from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from Utils.utils import read_pdf
from streamlit.runtime.uploaded_file_manager import UploadedFile

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def summarize(text: str) -> str:
    """
    Summarizes the given text using a pre-trained language model.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized version of the input text.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(**inputs)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_pdf(uploaded_file: UploadedFile) -> str:
    """
    Summarizes the content of a PDF file.

    Args:
        uploaded_file (UploadedFile): The PDF file to be summarized.

    Returns:
        str: The summarized version of the PDF content.
    """
    text = read_pdf(uploaded_file)
    if not text:
        return "No text could be extracted from the PDF."
    print(f"Extracted text from PDF: {text[:500]}...")  # Print the first 500 characters for debugging
    return summarize(text)
