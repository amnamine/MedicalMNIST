import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("medicalmnist.keras")
classes = ["AbdomenCT","BreastMRI","CXR","ChestCT","Hand","HeadCT"]

selected_image = None

# ================= WINDOW =================
root = tk.Tk()
root.title("Medical MNIST AI")
root.geometry("900x650")   # PERFECT for 1366x768 screens
root.configure(bg="#f0f4f8")

# ================= TITLE =================
title = tk.Label(root, text="Medical Image Classifier",
                 font=("Arial",22,"bold"), bg="#f0f4f8")
title.pack(pady=10)

# ================= IMAGE AREA =================
img_frame = tk.Frame(root, width=400, height=350, bg="white", bd=2, relief="solid")
img_frame.pack(pady=20)

img_label = tk.Label(img_frame, bg="white")
img_label.place(relx=0.5, rely=0.5, anchor="center")

# ================= RESULT TEXT =================
result_label = tk.Label(root,
                        text="Load an image to start",
                        font=("Arial",16,"bold"),
                        fg="blue", bg="#f0f4f8")
result_label.pack(pady=10)

# ================= FUNCTIONS =================
def load_image():
    global selected_image
    path = filedialog.askopenfilename()
    if path == "":
        return

    img = Image.open(path).resize((300,300))
    selected_image = img

    tk_img = ImageTk.PhotoImage(img)
    img_label.config(image=tk_img)
    img_label.image = tk_img

    result_label.config(text="Image loaded ✔")

def predict():
    global selected_image
    if selected_image is None:
        result_label.config(text="Load image first!", fg="red")
        return

    img = selected_image.resize((64,64))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    pred = model.predict(img, verbose=0)
    class_id = np.argmax(pred)
    conf = np.max(pred)*100

    result_label.config(text=f"{classes[class_id]}  ({conf:.2f}%)",
                        fg="green")

def reset():
    global selected_image
    selected_image = None
    img_label.config(image="")
    result_label.config(text="Load an image to start", fg="blue")

# ================= BUTTON AREA =================
btn_frame = tk.Frame(root, bg="#f0f4f8")
btn_frame.pack(pady=30)

load_btn = tk.Button(btn_frame, text="Load Image",
                     font=("Arial",14), width=15, height=2,
                     command=load_image)
load_btn.grid(row=0, column=0, padx=15)

predict_btn = tk.Button(btn_frame, text="Predict",
                        font=("Arial",14), width=15, height=2,
                        command=predict)
predict_btn.grid(row=0, column=1, padx=15)

reset_btn = tk.Button(btn_frame, text="Reset",
                      font=("Arial",14), width=15, height=2,
                      command=reset)
reset_btn.grid(row=0, column=2, padx=15)

# ================= RUN =================
root.mainloop()