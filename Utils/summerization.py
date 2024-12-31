# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Define the summarization function
def summarize(text):
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
