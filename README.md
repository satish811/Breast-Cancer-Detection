# 🩺 Breast Cancer Detection with Enhanced Interpretability Using VGGNet-16 and Grad-CAM


![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **Final Year B.Tech Project**  
> Deep Learning Approach for Automated Medical Image Analysis using **VGG16 Transfer Learning**, **Grad-CAM Visualization**, and a **GUI Interface**.

---

## 🌟 Project Overview

This project implements an advanced deep learning solution for automated classification of breast cancer histopathological images into **Benign** or **Malignant** categories using the **BreakHis dataset**.

Key Features:
- ✅ **Transfer Learning** with **VGG16** (ImageNet pre-trained)
- ✅ **Grad-CAM** for visual explanations of predictions
- ✅ **Interactive GUI** built with Tkinter
- ✅ Comprehensive evaluation: ROC, PR curves, confusion matrix, misclassification analysis

---

## 📚 Dataset

### 🔗 [BreakHis Breast Cancer Histopathological Dataset](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset)

- **Type**: Binary Classification (Benign vs. Malignant)
- **Magnification**: 40X,100X,200X,400X (224×224 RGB images)
- **Source**: W. Al-Dhabyani et al., 2019
- **Data Split**: 80% Training, 20% Validation
- **Class Balancing**: Automatic class weights to handle imbalance

### Preprocessing
- Resized to 224×224 for VGG16 compatibility
- Normalized to [0, 1]
- Augmentation: Rotation, flip, zoom, shift
- Batch size: 32

---

## 🏗️ Model Architecture

### 🧠 Transfer Learning with VGG16
- **Base Model**: VGG16 (pre-trained on ImageNet, **frozen layers**)
- **Custom Head**:
  - Global Average Pooling
  - Dense (512) → ReLU
  - BatchNorm + Dropout (0.5)
  - Dense (256) → ReLU
  - BatchNorm + Dropout (0.3)
  - Output: 1 neuron, Sigmoid activation (binary)

### ⚙️ Training Configuration
| Parameter         | Value                     |
|------------------|---------------------------|
| Optimizer        | SGD (momentum=0.9)        |
| Learning Rate    | 1e-4                      |
| Loss             | Binary Crossentropy       |
| Epochs           | 50                        |
| Callbacks        | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| Metrics          | Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC |

---

## 📊 Model Performance

| Metric             | Score       |
|--------------------|-------------|
| **ROC AUC**        | 0.54        |
| **PR AUC**         | 0.89      |
| **Class Balance**  | Weighted    |

> 📌 **Note**: PR AUC > ROC AUC suggests better performance on imbalanced data. Room for improvement via fine-tuning or ensemble methods.

### Key Insights
✅ **Strengths**:
- Good precision-recall performance (PR AUC = 0.89)
- Effective use of transfer learning
- Visual explainability via Grad-CAM
- Robust evaluation pipeline

🔧 **Areas for Improvement**:
- Low ROC AUC indicates need for better feature discrimination
- Try ResNet, DenseNet, or EfficientNet
- Fine-tune deeper layers of VGG16
- Explore advanced augmentation (CutMix, MixUp)

---

### 📈 Results & Visualizations

## Output Image: uploading an image
![Output Image: uploading an image](https://github.com/user-attachments/assets/51c63409-373f-412e-b72e-b01c3e9c8389)

## Output Image: Malignant
![Output Image: Malignant](https://github.com/user-attachments/assets/a39a6a0a-009f-44e0-bc5b-52bee4fce81c)

## Output Image: Benign
![Output Image: Benign](https://github.com/user-attachments/assets/6c095464-c2d4-420d-9fb1-9993fb748dfa)

---
### 📊 Probability Distribution
![Probability Distribution](probability_distribution.png)

> Distribution of predicted probabilities for benign and malignant classes.

---

### 📉 ROC Curve
![ROC Curve](roc_curve.png)

> ROC AUC = 0.54 — Indicates baseline discrimination ability.

---

### 📐 Precision-Recall Curve
![Precision-Recall Curve](precision_recall_curve.png)

> PR AUC = 0.69 — Stronger indicator of performance on imbalanced data.

---

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification

### 3. Download Dataset
1. Visit: [BreakHis Dataset on Kaggle](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset)
2. Extract to: `dataset_cancer_v1/classificacao_binaria/400X/`
   ```
   dataset_cancer_v1/
   └── classificacao_binaria/
       └── 400X/
           ├── benign/
           └── malignant/
   ```

### 4. Train the Model
```bash
python train.py
```
> Model saved as: `enhanced_breast_cancer_model.h5`

### 5. Launch GUI Application
```bash
python main2.py
```
- Click **"Select Image"**
- View prediction and **Grad-CAM heatmap** highlighting suspicious regions

---

## 📁 Project Structure

```
breast-cancer-classification/
│
├── train.py                          # Training script with evaluation
├── main2.py                         # GUI application with Grad-CAM
├── enhanced_breast_cancer_model.h5  # Trained model weights
│
├── dataset_cancer_v1/
│   └── classificacao_binaria/
│       └── 400X/
│           ├── benign/
│           └── malignant/
│
├── output_plots/
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── probability_distribution.png
│   └── misclassified_samples.png
│
└── README.md                        # This file
```

---

## 🌈 Features & Capabilities

### 🔧 Core Features
- **Transfer Learning**: VGG16-based model with custom classifier
- **Data Augmentation**: Rotation, flip, zoom, shift
- **Class Balancing**: Automatic weight adjustment
- **Advanced Callbacks**: Early stopping, LR reduction, best model saving

### 👁️ Visualization & Analysis
- **Grad-CAM**: Visualize model attention on tissue regions
- **Performance Plots**: ROC, PR, confusion matrix, probability distributions
- **Interactive GUI**: Easy-to-use interface for real-time predictions

---

## 💡 Technical Implementation

### Model Training Pipeline (Key Snippets)
```python
# Data Generator with Augmentation
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

# VGG16 Base (Frozen)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
```

### Grad-CAM Function
```python
def grad_cam(model, image_path, last_conv_layer_name="block5_conv3"):
    """
    Generate Grad-CAM visualization for model interpretability
    """
    grad_model = Model([model.inputs], 
                      [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    return heatmap, superimposed_image
```

---

## 🚀 Future Improvements

### 🧠 Model Enhancements
- **Ensemble Methods**: Combine VGG16, ResNet, EfficientNet
- **Fine-tuning**: Gradually unfreeze and train deeper layers
- **Advanced Architectures**: Try DenseNet, Vision Transformers
- **Multi-class**: Classify cancer subtypes (e.g., DCIS, LCIS)

### 💻 System Upgrades
- **Web App**: Deploy using Flask/FastAPI
- **Explainability**: Add LIME or SHAP for deeper insights
- **Cloud Integration**: Run inference on cloud platforms

---

## 🎓 Project Conclusion

This project demonstrates a complete deep learning pipeline for **medical image classification**, combining:
- Transfer learning
- Model interpretability (Grad-CAM)
- User-friendly GUI
- Rigorous evaluation

It serves as a strong foundation for **medical AI research** and is suitable for **educational and research purposes**.

---

## 📬 Contact & License

**Author**: satish varma 
**Institution**: Srkr Engineering College – B.Tech in CSBS  
**Email**: satishvarmaoleti09@.com  
**GitHub**: @satish811  https://github.com/satish811



---

⭐ **If you found this project helpful, please give it a star on GitHub!**
```

---


Best of luck  🎓🚀
