import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from deepface import DeepFace
import cv2
from collections import Counter
import os

def analyze_emotion_video(video_path, output_path="output_emotion_video.mp4", show_window=False):
    if not os.path.exists(video_path):
        print("❌ Video file not found!")
        return None

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"✅ Processing started...\nFPS: {fps}, Resolution: {width}x{height}")
    frame_count = 0
    emotions_detected = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % 5 != 0:  # process every 5th frame for speed
            out.write(frame)
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

            cv2.putText(frame, f"{dominant_emotion}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        except Exception as e:
            pass

        if show_window:
            cv2.imshow("Emotion Detection (DeepFace)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.write(frame)

        if len(emotions_detected) >= 20:  # Sufficient samples for fast analysis
            break

    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()

    if emotions_detected:
        dominant = Counter(emotions_detected).most_common(1)[0][0]
        print(f"\n🎬 Dominant Emotion: {dominant}")
        print(f"💾 Saved annotated video as: {output_path}")
        return dominant.lower()
    else:
        print("⚠️ No emotions detected confidently.")
        return None

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "Videos", "video01.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(base_dir, "Videos", "3712705-uhd_4096_2160_25fps.mp4")
    result = analyze_emotion_video(video_path)
    print("Detected emotion:", result)
