import cv2
import pickle
import shutil
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RAW_DB_DIR = BASE_DIR / "data" / "database" / "clean_db"
DB_PATH = BASE_DIR / "outputs" / "db_embeddings.pkl"
TEMP_PATH = BASE_DIR / "outputs" / "gui_temp.jpg"

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
THRESHOLD = 0.65

BG = "#020617"
CARD = "#0F172A"
CARD2 = "#111827"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
BLUE = "#2563EB"
GREEN = "#16A34A"
PURPLE = "#7C3AED"
RED = "#DC2626"
ORANGE = "#EA580C"

cap = None
running = False
db = []


def log(msg):
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)
    status_var.set(msg)


def load_db():
    global db
    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            db = pickle.load(f)
        log(f"Loaded database: {len(db)} embeddings")
    else:
        db = []
        log("Database not found. Please rebuild database.")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recognize_image(img_path):
    if not db:
        return "No database", 0

    rep = DeepFace.represent(
        img_path=str(img_path),
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False
    )

    emb = rep[0]["embedding"]
    best_name = "Unknown"
    best_score = -1

    for record in db:
        score = cosine_similarity(emb, record["embedding"])
        if score > best_score:
            best_score = score
            best_name = record["identity"]

    if best_score < THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


def safe_name(name):
    return "".join(c for c in name.strip() if c.isalnum() or c in (" ", "_", "-")).strip()


def run_command(command):
    log(f"> {command}")
    result = subprocess.run(command, cwd=SRC_DIR, shell=True, capture_output=True, text=True)
    if result.stdout:
        log(result.stdout.strip())
    if result.stderr:
        log(result.stderr.strip())


def rebuild_database():
    try:
        log("Rebuilding database...")
        run_command("python preprocess_images.py")
        run_command("python 01_build_embeddings.py")
        load_db()
        log("Database rebuilt successfully.")
        messagebox.showinfo("Success", "Database rebuilt successfully.")
    except Exception as e:
        log(f"Error: {e}")
        messagebox.showerror("Error", str(e))


def rebuild_now():
    threading.Thread(target=rebuild_database, daemon=True).start()


def add_new_person():
    files = filedialog.askopenfilenames(
        title="Select person images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )

    if not files:
        return

    name = simpledialog.askstring("Person Name", "Enter person's name:")
    if not name:
        return

    name = safe_name(name)
    if not name:
        messagebox.showerror("Error", "Invalid name.")
        return

    person_dir = RAW_DB_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(files, start=1):
        src = Path(file)
        dst = person_dir / f"{name}_{i}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    log(f"Added {len(files)} images for {name}")
    threading.Thread(target=rebuild_database, daemon=True).start()


def test_image():
    file = filedialog.askopenfilename(
        title="Select image to test",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )

    if not file:
        return

    def worker():
        try:
            name, score = recognize_image(file)
            result_var.set(f"{name}")
            score_var.set(f"Confidence Score: {score:.3f}")
            log(f"Image test result: {name} | {score:.4f}")
        except Exception as e:
            result_var.set("Error")
            score_var.set("Confidence Score: 0.000")
            log(f"Image test error: {e}")

    threading.Thread(target=worker, daemon=True).start()


def start_camera():
    global cap, running

    if running:
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Cannot open camera.")
        return

    running = True
    log("Camera started.")
    update_frame()


def stop_camera():
    global cap, running

    running = False

    if cap is not None:
        cap.release()
        cap = None

    video_label.configure(image="")
    result_var.set("Camera stopped")
    score_var.set("Confidence Score: --")
    log("Camera stopped.")


def update_frame():
    global cap, running

    if not running or cap is None:
        return

    ret, frame = cap.read()

    if not ret:
        stop_camera()
        return

    frame = cv2.flip(frame, 1)

    try:
        cv2.imwrite(str(TEMP_PATH), frame)
        name, score = recognize_image(TEMP_PATH)

        result_var.set(name)
        score_var.set(f"Confidence Score: {score:.3f}")

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (20, 20), (430, 80), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{name}  {score:.2f}",
            (35, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            3
        )

    except Exception:
        result_var.set("Recognizing...")
        score_var.set("Confidence Score: --")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    screen_w = root.winfo_screenwidth()
    video_w = int(screen_w * 0.50)
    video_h = int(video_w * 0.62)

    img = img.resize((video_w, video_h))
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(500, update_frame)


def exit_app():
    stop_camera()
    root.destroy()


root = tk.Tk()
root.title("Face Recognition System - Enterprise PRO")
root.configure(bg=BG)
root.state("zoomed")

status_var = tk.StringVar(value="System ready")
result_var = tk.StringVar(value="No result yet")
score_var = tk.StringVar(value="Confidence Score: --")

header = tk.Frame(root, bg=BG)
header.pack(fill="x", padx=36, pady=(22, 10))

tk.Label(
    header,
    text="Face Recognition System",
    font=("Segoe UI", 34, "bold"),
    fg=TEXT,
    bg=BG
).pack(anchor="w")

tk.Label(
    header,
    text="DeepFace • Facenet512 • Live Camera • Image Test • Add Person • Database Management",
    font=("Segoe UI", 13),
    fg=MUTED,
    bg=BG
).pack(anchor="w", pady=(4, 0))

main = tk.Frame(root, bg=BG)
main.pack(fill="both", expand=True, padx=36, pady=20)

left = tk.Frame(main, bg=CARD, width=330)
left.pack(side="left", fill="y", padx=(0, 18))
left.pack_propagate(False)

center = tk.Frame(main, bg=CARD, width=760)
center.pack(side="left", fill="both", expand=True, padx=(0, 18))

right = tk.Frame(main, bg=CARD, width=390)
right.pack(side="left", fill="y")
right.pack_propagate(False)

tk.Label(left, text="Controls", font=("Segoe UI", 22, "bold"), fg=TEXT, bg=CARD).pack(anchor="w", padx=24, pady=(28, 18))


def btn(parent, text, color, command):
    b = tk.Button(
        parent,
        text=text,
        font=("Segoe UI", 13, "bold"),
        bg=color,
        fg="white",
        activebackground=color,
        activeforeground="white",
        width=25,
        height=2,
        bd=0,
        cursor="hand2",
        command=command
    )
    b.pack(padx=24, pady=10)
    return b


btn(left, "Add New Person Images", BLUE, add_new_person)
btn(left, "Test Image", PURPLE, test_image)
btn(left, "Rebuild Database", ORANGE, rebuild_now)
btn(left, "Start Camera", GREEN, start_camera)
btn(left, "Stop Camera", RED, stop_camera)
btn(left, "Exit Application", "#334155", exit_app)

tk.Label(
    left,
    text="Workflow:\n1. Add person images\n2. Rebuild database\n3. Test image or start camera\n4. Review score and logs",
    font=("Segoe UI", 11),
    fg=MUTED,
    bg=CARD,
    wraplength=270,
    justify="left"
).pack(anchor="w", padx=24, pady=(28, 0))

tk.Label(center, text="Live Camera Preview", font=("Segoe UI", 22, "bold"), fg=TEXT, bg=CARD).pack(anchor="w", padx=24, pady=(24, 12))

video_frame = tk.Frame(center, bg="#020617")
video_frame.pack(fill="x", padx=24, pady=(0, 18))

video_label = tk.Label(video_frame, bg="#020617")
video_label.pack(padx=10, pady=10)

result_card = tk.Frame(center, bg=CARD2)
result_card.pack(fill="x", padx=24, pady=(5, 20))

tk.Label(
    result_card,
    text="Recognition Result",
    font=("Segoe UI", 14, "bold"),
    fg=MUTED,
    bg=CARD2
).pack(pady=(18, 2))

tk.Label(
    result_card,
    textvariable=result_var,
    font=("Segoe UI", 34, "bold"),
    fg="#22C55E",
    bg=CARD2
).pack(pady=(0, 4))

tk.Label(
    result_card,
    textvariable=score_var,
    font=("Segoe UI", 14),
    fg=MUTED,
    bg=CARD2
).pack(pady=(0, 18))

tk.Label(right, text="System Log", font=("Segoe UI", 22, "bold"), fg=TEXT, bg=CARD).pack(anchor="w", padx=20, pady=(24, 12))

log_box = tk.Text(
    right,
    bg="#020617",
    fg="#D1D5DB",
    insertbackground="white",
    font=("Consolas", 10),
    bd=0,
    wrap="word"
)
log_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

footer = tk.Frame(root, bg=BG)
footer.pack(fill="x", padx=36, pady=(0, 14))

tk.Label(
    footer,
    textvariable=status_var,
    font=("Segoe UI", 11, "bold"),
    fg="#22C55E",
    bg=BG
).pack(side="left")

tk.Label(
    footer,
    text="Khabbab Face Recognition Project | Enterprise GUI",
    font=("Segoe UI", 10),
    fg=MUTED,
    bg=BG
).pack(side="right")

load_db()
root.protocol("WM_DELETE_WINDOW", exit_app)
root.mainloop()