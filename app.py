import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
import pandas as pd
import datetime
import os
from predict import predict_suicidal, prepare_chat_data, fuse_predictions
from speech_to_text import audio_file_to_text
from Emotion_detection import analyze_emotion_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

# ===============================
# Utility functions
# ===============================

def classify_video(video_path):
    result = analyze_emotion_video(video_path, output_path=os.path.join(BASE_DIR, "output_emotion_video.mp4"), show_window=False)
    suicidal_emotions = {"sad", "angry", "fear", "disgust"}
    non_suicidal_emotions = {"neutral", "happy", "surprise"}

    if result in suicidal_emotions:
        return "Suicidal", 0.8
    elif result in non_suicidal_emotions:
        return "Non-Suicidal", 0.3
    else:
        return "Non-Suicidal", 0.5


def save_daily_result(date, label, prob):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame([[date, label, prob]], columns=["Date", "Prediction", "Probability"])
    file_path = os.path.join(RESULTS_DIR, "daily_results.csv")
    df.to_csv(file_path, mode="a", index=False, header=not os.path.exists(file_path))


def generate_weekly_summary():
    file_path = os.path.join(RESULTS_DIR, "daily_results.csv")
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    if df.empty or 'Date' not in df.columns:
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    last_week = df[df['Date'] >= (datetime.datetime.now() - datetime.timedelta(days=7))]

    if len(last_week) == 0:
        return None

    suicidal_days = (last_week['Prediction'] == "Suicidal").sum()
    non_suicidal_days = (last_week['Prediction'] == "Non-Suicidal").sum()
    avg_prob = last_week['Probability'].mean()

    summary = pd.DataFrame({
        "Total_Inputs": [len(last_week)],
        "Suicidal_Days": [suicidal_days],
        "Non_Suicidal_Days": [non_suicidal_days],
        "Average_Suicide_Probability": [round(avg_prob, 3)]
    })
    summary.to_csv(os.path.join(RESULTS_DIR, "weekly_summary.csv"), index=False)
    return summary


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Suicide Detection Dashboard", layout="centered")

st.title("🧠 Suicide Risk Detection Dashboard")

st.header("🔹 Daily Input")
text_file = st.file_uploader("Upload Text File (.txt)", type=["txt"])
audio_file = st.file_uploader("Upload Audio File (.wav)", type=["wav"])
video_file = st.file_uploader("Upload Video File (.mp4)", type=["mp4"])

if st.button("🔍 Analyze Today's Input"):
    if not text_file and not audio_file and not video_file:
        st.warning("⚠️ Please upload at least one file (Text, Audio, or Video) to analyze.")
    else:
        text_out = voice_out = video_out = None

        with st.spinner("Analyzing input data..."):
            # Text Model
            if text_file:
                text_data = text_file.read().decode("utf-8")
                prediction, score = predict_suicidal(text_data)
                text_out = (prediction, score)
                st.write(f"📘 Text Model: **{prediction} ({score:.2f})**")

            # Voice Model
            if audio_file:
                temp_audio_path = os.path.join(BASE_DIR, "temp_audio.wav")
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_file.read())
                voice_text = audio_file_to_text(temp_audio_path)
                prediction, score = predict_suicidal(voice_text)
                voice_out = (prediction, score)
                st.write(f"🎤 Voice Model: **{prediction} ({score:.2f})**")

            # Video Model
            if video_file:
                temp_video_path = os.path.join(BASE_DIR, "temp_video.mp4")
                with open(temp_video_path, "wb") as f:
                    f.write(video_file.read())
                label, score = classify_video(temp_video_path)
                video_out = (label, score)
                st.write(f"🎥 Video Model: **{label} ({score:.2f})**")

            # Combine Predictions
            final_label, final_prob = fuse_predictions(text_out, voice_out, video_out)
            st.subheader("🧩 Final Fusion Result")
            if final_label == "Suicidal":
                st.error(f"**Final Prediction:** {final_label}")
            else:
                st.success(f"**Final Prediction:** {final_label}")
            st.write(f"**Suicidal Probability:** {final_prob:.2f}")

            # Save daily result
            save_daily_result(datetime.datetime.now().strftime("%Y-%m-%d"), final_label, round(final_prob, 3))
            st.info("✅ Daily result saved successfully!")


st.divider()
st.header("📅 Weekly Summary")

if st.button("📊 Generate Weekly Report"):
    summary = generate_weekly_summary()
    if summary is not None:
        st.write(summary)
    else:
        st.warning("No data available for the last 7 days.")
