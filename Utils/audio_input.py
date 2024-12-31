import speech_recognition as sr
import tempfile

def listen():
    """
    Listens to audio input from the microphone and attempts to recognize the speech using Google's speech recognition API.

    Returns:
        str: The recognized text from the audio input, or an error message if the recognition fails.
        str: The path to the temporary audio file.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:  # Use the default microphone as the audio source
        print("Listening...")
        audio = recognizer.listen(source)  # Listen for speech
        try:
            recognized_text = recognizer.recognize_google(audio)  # Recognize the speech using Google's speech recognition API
            print(recognized_text)  # Print the recognized text

            # Save the audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(audio.get_wav_data())
                temp_audio_path = temp_audio_file.name

            return recognized_text, temp_audio_path
        except sr.UnknownValueError:  # Handle the exception of unintelligible speech
            error_message = "Sorry, I could not understand the audio."
            print(error_message)
            return error_message, None
        except sr.RequestError:  # Handle the exception of request error
            error_message = "Request failed, please try again."
            print(error_message)
            return error_message, None