import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.cm as cm

# Set dataset path
dataset_path = "dataset_cancer_v1/classificacao_binaria/400X"
img_size = 224
batch_size = 32
epochs = 50 # Increased epochs with early stopping

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)
# Automatically compute class weights based on training data
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight = dict(enumerate(class_weights))
print("Class weights:", class_weight)

val_gen = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Build enhanced CNN Model using Transfer Learning
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
for layer in base_model.layers:
    layer.trainable = False  


x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Custom metrics
def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1-y_true) * y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1-y_pred), 'float'), axis=0)
    
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    f1 = 2*precision*recall / (precision+recall+tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)

# Compile with additional metrics
from tensorflow.keras.optimizers import SGD

optimizer = SGD(learning_rate=1e-4, momentum=0.9)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score]
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight
)

# Save the trained model
model.save("enhanced_breast_cancer_model.h5")
print("✅ Enhanced model saved as enhanced_breast_cancer_model.h5")

# Evaluate on validation set
val_gen.reset()
y_pred = model.predict(val_gen)
y_pred_classes = (y_pred > 0.5).astype("int32")
y_true = val_gen.classes

y_pred_probs = y_pred.flatten()
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
)

# Create output directory
os.makedirs("output_plots", exist_ok=True)

# --- 1. Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}", color='darkorange')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("output_plots/precision_recall_curve.png", dpi=300)
plt.show()

# --- 2. ROC Curve ---
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = roc_auc_score(y_true, y_pred_probs)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}", color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("output_plots/roc_curve.png", dpi=300)
plt.show()

# --- 3. Probability Distribution ---
plt.figure(figsize=(8, 4))
sns.histplot(y_pred_probs[y_true == 0], bins=25, color='blue', label='Benign', kde=True)
sns.histplot(y_pred_probs[y_true == 1], bins=25, color='red', label='Malignant', kde=True)
plt.title("Predicted Probability Distributions")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("output_plots/probability_distribution.png", dpi=300)
plt.show()

# --- 4. Misclassified Image Viewer ---
file_paths = val_gen.filepaths
misclassified_indices = np.where(y_true != y_pred_classes)[0]
sample_indices = random.sample(list(misclassified_indices), min(5, len(misclassified_indices)))

plt.figure(figsize=(15, 6))
for i, idx in enumerate(sample_indices):
    img = cv2.imread(file_paths[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    true_label = "Benign" if y_true[idx] == 0 else "Malignant"
    pred_label = "Benign" if y_pred_classes[idx] == 0 else "Malignant"
    
    plt.subplot(1, len(sample_indices), i + 1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
    plt.axis('off')

plt.suptitle("Misclassified Samples", fontsize=16)
plt.tight_layout()
plt.savefig("output_plots/misclassified_samples.png", dpi=300)
plt.show()


# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=["Benign", "Malignant"]))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Benign", "Malignant"], 
            yticklabels=["Benign", "Malignant"])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

# Grad-CAM function (enhanced with prediction confidence)
def grad_cam(model, image_path, last_conv_layer_name="block5_conv3"):
    img_array = load_image(image_path)
    
    # Get prediction confidence
    pred = model.predict(img_array)[0][0]
    class_index = int(pred > 0.5)
    confidence = pred if class_index == 1 else 1 - pred
    
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()[0]
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Superimpose heatmap
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image\nPrediction: {'Malignant' if class_index else 'Benign'} ({confidence:.2%})")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")
    
    plt.show()
    
    # Text explanation
    print("\n🔍 **Grad-CAM Explanation:**")
    print(f"Prediction confidence: {confidence:.2%}")
    if class_index == 1:
        print("🔥 The highlighted red regions correspond to features the model considers most indicative of malignancy.")
    else:
        print("✅ The blue regions indicate areas the model used to determine this is benign tissue.")
    print("The heatmap shows where the model 'looked' to make its decision.")

# Test Grad-CAM on a sample image
sample_image_path = "dataset_cancer_v1/classificacao_binaria/100X/benign/SOB_B_A-14-22549AB-40-001.png"
grad_cam(model, sample_image_path)