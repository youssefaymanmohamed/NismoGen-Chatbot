import pyttsx3

def speak(text):
    """
    Convert the given text to speech and play it.

    Args:
        text (str): The text to be converted to speech.
    """
    engine = pyttsx3.init() # Initialize the Pyttsx3 engine
    engine.say(text)    # Convert the text to speech
    engine.runAndWait() # Play the speech