# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")

# Define the image captioning function
def query(image):
    """
    Generates a caption for the given image using a pre-trained model.

    Args:
        image (PIL.Image or torch.Tensor): The input image to be captioned.

    Returns:
        str: The generated caption for the image.
    """
    inputs = processor(image, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    caption = model.generate(**inputs)
    return processor.decode(caption[0], skip_special_tokens=True)
