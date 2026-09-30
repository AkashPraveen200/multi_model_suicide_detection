import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from deepface import DeepFace
import speech_recognition as sr
import os
import cv2
from collections import Counter

# ============================
# 1️⃣ Load Electra Model
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "TRAINING", "MODELS")

tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH)
model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
model.eval()

# ============================
# 2️⃣ Text Prediction
# ============================
def prepare_chat_data(file_path):
    """Reads .txt file and returns single cleaned string"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = " ".join(line.strip() for line in file if line.strip())
        return text
    except Exception as e:
        print(f"❌ Error reading text: {e}")
        return ""

def predict_suicidal(text: str):
    """Predict suicide probability from text using Electra"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs).item()
            suicide_score = probs[0][1].item()

        label = "Suicidal" if pred == 1 else "Non-Suicidal"
        return label, round(suicide_score, 4)

    except Exception as e:
        print(f"❌ Text prediction error: {e}")
        return "Unknown", 0.0

# ============================
# 3️⃣ Voice (Speech → Text → Electra)
# ============================
def audio_file_to_text(file_path):
    """Convert voice to text using SpeechRecognition"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        print(f"❌ Speech-to-text error: {e}")
        return ""

def predict_voice(file_path):
    """Predict suicide risk from voice file"""
    text = audio_file_to_text(file_path)
    if not text:
        return "Non-Suicidal", 0.0
    return predict_suicidal(text)

# ============================
# 4️⃣ Video Emotion Detection
# ============================
def analyze_emotion_video(video_path):
    """Detect dominant emotion using DeepFace"""
    if not os.path.exists(video_path):
        print("❌ Video not found.")
        return None

    cap = cv2.VideoCapture(video_path)
    frame_count, emotions_detected = 0, []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 != 0:
            continue

        try:
            h, w = frame.shape[:2]
            if w > 640:
                frame_resized = cv2.resize(frame, (640, int(640 * h / w)))
            else:
                frame_resized = frame

            result = DeepFace.analyze(frame_resized, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            emotions_detected.append(dominant_emotion)
        except Exception:
            continue

        if len(emotions_detected) >= 20:  # 20 emotion samples are sufficient for fast analysis
            break

    cap.release()
    if emotions_detected:
        dominant = Counter(emotions_detected).most_common(1)[0][0].lower()
        return dominant
    return None

def predict_video(video_path):
    """Convert emotion into suicide likelihood"""
    emotion = analyze_emotion_video(video_path)
    suicidal_emotions = {"sad", "angry", "fear", "disgust"}
    non_suicidal_emotions = {"neutral", "happy", "surprise"}

    if emotion in suicidal_emotions:
        return "Suicidal", 0.8
    elif emotion in non_suicidal_emotions:
        return "Non-Suicidal", 0.3
    else:
        return "Non-Suicidal", 0.5

# ============================
# 5️⃣ Fusion Algorithm
# ============================
def fuse_predictions(text_output=None, voice_output=None, video_output=None):
    """
    Combine model outputs dynamically.
    Weighted: Text=0.4, Voice=0.4, Video=0.2
    Adjusts automatically if inputs missing.
    """
    weights = {
        "text": 0.4 if text_output else 0.0,
        "voice": 0.4 if voice_output else 0.0,
        "video": 0.2 if video_output else 0.0
    }

    total_weight = sum(weights.values())
    if total_weight == 0:
        return "Unknown", 0.0

    for key in weights:
        weights[key] /= total_weight

    text_prob = text_output[1] if text_output else None
    voice_prob = voice_output[1] if voice_output else None
    video_prob = video_output[1] if video_output else None

    final_prob = 0.0
    if text_prob is not None:
        final_prob += text_prob * weights["text"]
    if voice_prob is not None:
        final_prob += voice_prob * weights["voice"]
    if video_prob is not None:
        final_prob += video_prob * weights["video"]

    final_label = "Suicidal" if final_prob >= 0.5 else "Non-Suicidal"

    print("\n=== Fusion Summary ===")
    if text_output: print(f"Text Model:  {text_output[0]} ({text_output[1]:.2f}) [w={weights['text']:.2f}]")
    if voice_output: print(f"Voice Model: {voice_output[0]} ({voice_output[1]:.2f}) [w={weights['voice']:.2f}]")
    if video_output: print(f"Video Model: {video_output[0]} ({video_output[1]:.2f}) [w={weights['video']:.2f}]")
    print("-------------------------")
    print(f"Final Prediction: {final_label}")
    print(f"Suicidal Probability: {final_prob:.4f}")

    return final_label, round(final_prob, 4)

# ============================
# 6️⃣ Example Run
# ============================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    text_file = os.path.join(BASE_DIR, "Texts", "sucide1.txt")
    audio_file = os.path.join(BASE_DIR, "Audios", "sucide audio 1.wav")
    video_file = os.path.join(BASE_DIR, "Videos", "video01.mp4")
    if not os.path.exists(video_file):
        video_file = os.path.join(BASE_DIR, "Videos", "3712705-uhd_4096_2160_25fps.mp4")

    text_data = prepare_chat_data(text_file)
    text_out = predict_suicidal(text_data)
    voice_out = predict_voice(audio_file)
    video_out = predict_video(video_file)

    label, score = fuse_predictions(text_out, voice_out, video_out)
    print("\nFinal Output:")
    print("Prediction:", label)
    print("Suicidal Probability:", score)
