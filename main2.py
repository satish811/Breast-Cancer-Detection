import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk

# Load trained model
model = load_model("enhanced_breast_cancer_model.h5")
img_size = 224

# Function to load and preprocess image
def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"⚠️ Image not found: {img_path}")
    
    img = cv2.resize(img, (img_size, img_size))
    img_array = img / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0), img

import tensorflow as tf
import numpy as np
import cv2

def grad_cam(img_path, last_conv_layer_name="block5_conv3"):
    img_array, orig_img = load_image(img_path)

    # Model to extract features from last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predictions = tf.convert_to_tensor(predictions)  # Ensure it is a Tensor
        loss = predictions[0, 0]  # Access first batch element

    grads = tape.gradient(loss, conv_outputs)  # Now it works!
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()[0]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Superimpose heatmap
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

    return int(loss.numpy() > 0.5), superimposed_img  # Convert loss to class index





# Function to classify image
def classify_image():
    global img_label, heatmap_label, result_label

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    
    if not file_path:
        return

    # Predict and generate heatmap
    class_index, heatmap = grad_cam(file_path)

    # Convert original and heatmap images to display in Tkinter
    orig_img = Image.open(file_path)
    orig_img = orig_img.resize((300, 300))
    orig_img_tk = ImageTk.PhotoImage(orig_img)

    heatmap_img = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    heatmap_img = heatmap_img.resize((300, 300))
    heatmap_img_tk = ImageTk.PhotoImage(heatmap_img)

    # Update labels with images
    img_label.config(image=orig_img_tk)
    img_label.image = orig_img_tk

    heatmap_label.config(image=heatmap_img_tk)
    heatmap_label.image = heatmap_img_tk

    # Classification result
    diagnosis = "Malignant 🔴" if class_index == 1 else "Benign 🟢"
    explanation = (
        "🔥 The highlighted regions show tumor-related features, indicating malignancy."
        if class_index == 1 else
        "✅ The model found no critical tumor regions, suggesting a benign case."
    )

    result_label.config(text=f"Prediction: {diagnosis}\n{explanation}", font=("Arial", 12, "bold"), fg="blue")

# Tkinter GUI setup
root = tk.Tk()
root.title("Breast Cancer Classification")
root.geometry("700x600")
root.configure(bg="white")

# Upload button
upload_btn = tk.Button(root, text="Select Image", command=classify_image, font=("Arial", 14), bg="lightblue")
upload_btn.pack(pady=10)

# Labels for displaying images
img_label = tk.Label(root)
img_label.pack()

heatmap_label = tk.Label(root)
heatmap_label.pack()

# Classification result
result_label = tk.Label(root, text="", font=("Arial", 12), wraplength=600, justify="center")
result_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
