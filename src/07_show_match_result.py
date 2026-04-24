import pickle
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "outputs" / "db_embeddings.pkl"
QUERY_IMG = BASE_DIR / "data" / "query_distorted" / "test1.jpg"

MODEL_NAME = "Facenet512"
THRESHOLD = 0.70  # يمكنك تغييره لاحقًا

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def read_image_rgb(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    if not DB_PATH.exists():
        print(f"[ERROR] Embeddings file not found: {DB_PATH}")
        return

    if not QUERY_IMG.exists():
        print(f"[ERROR] Query image not found: {QUERY_IMG}")
        return

    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)

    if not db:
        print("[ERROR] Database embeddings are empty.")
        return

    query_rep = DeepFace.represent(
        img_path=str(QUERY_IMG),
        model_name="Facenet",
        enforce_detection=False
    )[0]["embedding"]

    best_score = -1
    best_identity = None
    best_match_path = None

    for entry in db:
        score = cosine_similarity(query_rep, entry["embedding"])

        if score > best_score:
            best_score = score
            best_identity = entry["identity"]
            best_match_path = entry["path"]

    print("\n===== MATCH RESULT =====")
    print(f"Query image : {QUERY_IMG.name}")
    print(f"Best match  : {best_identity}")
    print(f"Score       : {best_score:.4f}")

    if best_score < THRESHOLD:
        print("Decision    : Unknown person")
    else:
        print(f"Decision    : Matched with {best_identity}")

    # عرض الصور
    query_img = read_image_rgb(QUERY_IMG)
    match_img = read_image_rgb(best_match_path)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(query_img)
    plt.title("Query Distorted Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(match_img)
    plt.title(f"Best Match: {best_identity}\nScore: {best_score:.4f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()