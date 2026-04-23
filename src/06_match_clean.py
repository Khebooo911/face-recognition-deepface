import pickle
import numpy as np
from deepface import DeepFace

DB_PATH = "../outputs/db_embeddings.pkl"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_match(query_img_path):
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)

    rep = DeepFace.represent(
        img_path=query_img_path,
        model_name="Facenet",
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

    return best_match, best_score


if __name__ == "__main__":
    query_img = "../data/database/clean_db/Khabbab/nomask_01.jpg"

    match, score = find_match(query_img)

    print("\n===== CLEAN RESULT =====")
    print(f"Best match: {match['identity']}")
    print(f"Score: {score}")