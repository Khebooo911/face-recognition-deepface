import cv2
import pickle
import numpy as np
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk
from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "outputs" / "db_embeddings.pkl"

MODEL_NAME = "Facenet512"
THRESHOLD = 0.65

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_db():
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)

db = load_db()

cap = None
running = False

def start_camera():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    update_frame()

def stop_camera():
    global running
    running = False
    if cap:
        cap.release()

def recognize(frame):
    temp_path = BASE_DIR / "outputs" / "temp.jpg"
    cv2.imwrite(str(temp_path), frame)

    try:
        rep = DeepFace.represent(
            img_path=str(temp_path),
            model_name=MODEL_NAME,
            enforce_detection=False
        )

        emb = rep[0]["embedding"]

        best_name = "Unknown"
        best_score = -1

        for r in db:
            score = cosine_similarity(emb, r["embedding"])
            if score > best_score:
                best_score = score
                best_name = r["identity"]

        if best_score < THRESHOLD:
            best_name = "Unknown"

        return best_name, best_score
    except:
        return "Error", 0

def update_frame():
    global running

    if not running:
        return

    ret, frame = cap.read()

    if ret:
        name, score = recognize(frame)

        text = f"{name} ({score:.2f})"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        result_label.config(text=f"{name} | {score:.2f}")

    root.after(30, update_frame)

# UI
root = tk.Tk()
root.title("Face Recognition PRO")
root.geometry("900x600")

video_label = tk.Label(root)
video_label.pack()

result_label = tk.Label(root, text="Result", font=("Arial", 20, "bold"))
result_label.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Start Camera", bg="green", fg="white",
          command=start_camera, width=15).grid(row=0, column=0, padx=10)

tk.Button(btn_frame, text="Stop Camera", bg="red", fg="white",
          command=stop_camera, width=15).grid(row=0, column=1, padx=10)

root.mainloop()