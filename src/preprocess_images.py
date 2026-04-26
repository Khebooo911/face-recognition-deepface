from pathlib import Path
from PIL import Image

INPUT_DIR = Path("../data/database/clean_db")
OUTPUT_DIR = Path("../data/database/clean_db_processed")

TARGET_SIZE = (224, 224)

def process_image(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(TARGET_SIZE)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path.with_suffix(".jpg"), "JPEG", quality=95)

        print(f"[OK] {img_path}")
    except Exception as e:
        print(f"[ERROR] {img_path} -> {e}")

def main():
    all_images = list(INPUT_DIR.rglob("*.*"))

    for img_path in all_images:
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            relative_path = img_path.relative_to(INPUT_DIR)
            save_path = OUTPUT_DIR / relative_path
            process_image(img_path, save_path)

    print("\n[DONE] All images processed.")

if __name__ == "__main__":
    main()