import pandas as pd
import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
DAILY_FILE = os.path.join(RESULTS_DIR, "daily_results.csv")
WEEKLY_FILE = os.path.join(RESULTS_DIR, "weekly_summary.csv")

def save_daily_result(date, label, prob):
    """
    Saves the daily prediction result to Results/daily_results.csv
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame([[date, label, prob]], columns=["Date", "Prediction", "Probability"])
    df.to_csv(DAILY_FILE, mode="a", index=False, header=not os.path.exists(DAILY_FILE))


def generate_weekly_summary():
    """
    Reads last 7 days of data and generates a summary CSV.
    """
    if not os.path.exists(DAILY_FILE):
        return None

    df = pd.read_csv(DAILY_FILE)
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
    summary.to_csv(WEEKLY_FILE, index=False)
    return summary
