import pickle
from pathlib import Path
from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "database" / "clean_db_processed"
OUTPUT_PATH = BASE_DIR / "outputs" / "db_embeddings.pkl"

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"

def get_all_images(root_dir):
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(root_dir.rglob(ext))
    return image_paths

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_images = get_all_images(DB_DIR)
    print(f"[INFO] Found {len(all_images)} images")

    db_embeddings = []

    for img_path in all_images:
        try:
            rep = DeepFace.represent(
                img_path=str(img_path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )

            embedding = rep[0]["embedding"]

            db_embeddings.append({
                "path": str(img_path),
                "identity": img_path.parent.name,
                "embedding": embedding
            })

            print(f"[OK] {img_path}")

        except Exception as e:
            print(f"[ERROR] {img_path} -> {e}")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(db_embeddings, f)

    print(f"[DONE] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()