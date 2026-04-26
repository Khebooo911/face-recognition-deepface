import cv2
import pickle
import numpy as np
from pathlib import Path
from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "outputs" / "db_embeddings.pkl"

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
THRESHOLD = 0.65

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_database():
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)

def find_match(frame_path, db):
    rep = DeepFace.represent(
        img_path=str(frame_path),
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False
    )

    query_embedding = rep[0]["embedding"]

    best_match = None
    best_score = -1

    for record in db:
        score = cosine_similarity(query_embedding, record["embedding"])

        if score > best_score:
            best_score = score
            best_match = record

    if best_score < THRESHOLD:
        return "Unknown", best_score

    return best_match["identity"], best_score

def main():
    db = load_database()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    temp_path = BASE_DIR / "outputs" / "camera_frame.jpg"

    frame_count = 0
    current_name = "Waiting..."
    current_score = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 30 == 0:
            cv2.imwrite(str(temp_path), frame)

            try:
                current_name, current_score = find_match(temp_path, db)
                print(current_name, current_score)
            except:
                current_name = "Error"

        text = f"{current_name} | {current_score:.2f}"

        cv2.putText(
            frame,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()