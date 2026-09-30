import speech_recognition as sr
import os

def audio_file_to_text(file_path):
    """
    Convert a .wav audio file into text.
    Returns recognized text string or empty string if failed.
    """
    if not os.path.exists(file_path):
        print(f"❌ Audio file not found: {file_path}")
        return ""

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand the audio")
        return ""
    except sr.RequestError as e:
        print(f"❌ API request error: {e}")
        return ""
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""
