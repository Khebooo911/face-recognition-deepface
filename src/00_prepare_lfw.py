from pathlib import Path
from sklearn.datasets import fetch_lfw_people
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
LFW_TARGET = BASE_DIR / "data" / "database" / "lfw"

MAX_PEOPLE = 30
IMAGES_PER_PERSON = 2

def main():
    LFW_TARGET.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading LFW dataset...")
    lfw = fetch_lfw_people(min_faces_per_person=2, resize=1.0, color=True)

    images = lfw.images
    target = lfw.target
    names = lfw.target_names

    saved_count = 0
    used_people = 0
    per_person_counter = {}

    for idx, person_id in enumerate(target):
        person_name = names[person_id].replace(" ", "_")

        if person_name not in per_person_counter:
            if used_people >= MAX_PEOPLE:
                continue
            per_person_counter[person_name] = 0
            used_people += 1

        if per_person_counter[person_name] >= IMAGES_PER_PERSON:
            continue

        person_dir = LFW_TARGET / person_name
        person_dir.mkdir(parents=True, exist_ok=True)

        img = images[idx]
        img_path = person_dir / f"{person_name}_{per_person_counter[person_name] + 1}.jpg"
        Image.fromarray(img.astype("uint8")).save(img_path)

        per_person_counter[person_name] += 1
        saved_count += 1

    print(f"[DONE] Saved {saved_count} images for {used_people} people into: {LFW_TARGET}")

if __name__ == "__main__":
    main()