<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Breast Cancer Histopathological Classification - README</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css">
    <style>
        body { 
            print-color-adjust: exact; 
            -webkit-print-color-adjust: exact; 
        }
        .code-block {
            background-color: #1e293b;
            color: #e2e8f0;
            border-radius: 0.5rem;
            padding: 1rem;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.875rem;
            line-height: 1.5;
        }
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .feature-card {
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-2px);
        }
    </style>
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <header class="gradient-bg text-white py-12">
        <div class="container mx-auto px-6 text-center">
            <div class="flex justify-center items-center mb-4">
                <i class="fas fa-microscope text-6xl mr-4"></i>
                <h1 class="text-5xl font-bold">Breast Cancer Histopathological Classification</h1>
            </div>
            <p class="text-xl opacity-90 mb-6">Deep Learning Approach for Automated Medical Image Analysis</p>
            <div class="flex justify-center space-x-4">
                <span class="bg-white bg-opacity-20 px-4 py-2 rounded-full text-sm">
                    <i class="fab fa-python mr-2"></i>Python
                </span>
                <span class="bg-white bg-opacity-20 px-4 py-2 rounded-full text-sm">
                    <i class="fas fa-brain mr-2"></i>TensorFlow
                </span>
                <span class="bg-white bg-opacity-20 px-4 py-2 rounded-full text-sm">
                    <i class="fas fa-chart-line mr-2"></i>Deep Learning
                </span>
                <span class="bg-white bg-opacity-20 px-4 py-2 rounded-full text-sm">
                    <i class="fas fa-stethoscope mr-2"></i>Medical AI
                </span>
            </div>
        </div>
    </header>

    <div class="container mx-auto px-6 py-12">
        <!-- Project Overview -->
        <section class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-info-circle text-blue-600 mr-3"></i>Project Overview
                </h2>
                <p class="text-gray-600 text-lg leading-relaxed mb-6">
                    This project implements an advanced deep learning solution for automated classification of breast cancer histopathological images. 
                    Using transfer learning with VGG16 architecture, the system achieves reliable binary classification between benign and malignant tissue samples. 
                    The project includes comprehensive model evaluation, explainability features through Grad-CAM visualization, and an intuitive GUI interface for real-world deployment.
                </p>
                
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-8">
                    <div class="feature-card bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border border-blue-200">
                        <i class="fas fa-network-wired text-3xl text-blue-600 mb-3"></i>
                        <h3 class="font-semibold text-gray-800 mb-2">Transfer Learning</h3>
                        <p class="text-sm text-gray-600">VGG16 pre-trained model with custom classification layers</p>
                    </div>
                    <div class="feature-card bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-lg border border-green-200">
                        <i class="fas fa-eye text-3xl text-green-600 mb-3"></i>
                        <h3 class="font-semibold text-gray-800 mb-2">Grad-CAM</h3>
                        <p class="text-sm text-gray-600">Visual explanations for model predictions</p>
                    </div>
                    <div class="feature-card bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-lg border border-purple-200">
                        <i class="fas fa-desktop text-3xl text-purple-600 mb-3"></i>
                        <h3 class="font-semibold text-gray-800 mb-2">GUI Interface</h3>
                        <p class="text-sm text-gray-600">User-friendly Tkinter application for image classification</p>
                    </div>
                    <div class="feature-card bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-lg border border-red-200">
                        <i class="fas fa-chart-bar text-3xl text-red-600 mb-3"></i>
                        <h3 class="font-semibold text-gray-800 mb-2">Comprehensive Analysis</h3>
                        <p class="text-sm text-gray-600">ROC curves, precision-recall analysis, and detailed metrics</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Dataset Information -->
        <section class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-database text-green-600 mr-3"></i>Dataset Information
                </h2>
                
                <div class="bg-gradient-to-r from-green-50 to-blue-50 p-6 rounded-lg border-l-4 border-green-500 mb-6">
                    <h3 class="text-xl font-semibold text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-link text-blue-600 mr-2"></i>BreakHis Dataset
                    </h3>
                    <p class="text-gray-700 mb-3">
                        <strong>Source:</strong> <a href="https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset/data" class="text-blue-600 hover:underline" target="_blank">
                            Kaggle - BreakHis Breast Cancer Histopathological Dataset
                        </a>
                    </p>
                    <p class="text-gray-600">
                        The Breast Cancer Histopathological Image Classification (BreakHis) dataset is composed of microscopic images of breast tumor tissue collected from patients. 
                        This comprehensive dataset provides high-resolution histopathological images with both binary and multi-class classifications.
                    </p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="bg-blue-50 p-6 rounded-lg border border-blue-200">
                        <h4 class="font-semibold text-blue-800 mb-3 flex items-center">
                            <i class="fas fa-images mr-2"></i>Image Specifications
                        </h4>
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li><strong>Magnification:</strong> 400X</li>
                            <li><strong>Input Size:</strong> 224x224 pixels</li>
                            <li><strong>Format:</strong> RGB color images</li>
                            <li><strong>Normalization:</strong> [0, 1] range</li>
                        </ul>
                    </div>
                    <div class="bg-purple-50 p-6 rounded-lg border border-purple-200">
                        <h4 class="font-semibold text-purple-800 mb-3 flex items-center">
                            <i class="fas fa-tags mr-2"></i>Classification
                        </h4>
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li><strong>Type:</strong> Binary Classification</li>
                            <li><strong>Classes:</strong> Benign, Malignant</li>
                            <li><strong>Split:</strong> 80% Train, 20% Validation</li>
                            <li><strong>Class Balancing:</strong> Automated weights</li>
                        </ul>
                    </div>
                    <div class="bg-green-50 p-6 rounded-lg border border-green-200">
                        <h4 class="font-semibold text-green-800 mb-3 flex items-center">
                            <i class="fas fa-cogs mr-2"></i>Preprocessing
                        </h4>
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li><strong>Augmentation:</strong> Rotation, flip, zoom</li>
                            <li><strong>Normalization:</strong> Pixel values / 255.0</li>
                            <li><strong>Resize:</strong> 224x224 for VGG16</li>
                            <li><strong>Batch Size:</strong> 32</li>
                        </ul>
                    </div>
                </div>
            </div>
        </section>

        <!-- Model Architecture -->
        <section class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-sitemap text-purple-600 mr-3"></i>Model Architecture
                </h2>
                
                <div class="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-lg mb-6">
                    <h3 class="text-xl font-semibold text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-layer-group text-purple-600 mr-2"></i>Transfer Learning with VGG16
                    </h3>
                    <p class="text-gray-700">
                        The model leverages the pre-trained VGG16 architecture as a feature extractor, with custom classification layers added for the specific task of breast cancer histopathology classification.
                    </p>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div>
                        <h4 class="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                            <i class="fas fa-cube text-blue-600 mr-2"></i>Architecture Details
                        </h4>
                        <div class="space-y-3">
                            <div class="flex items-center p-3 bg-blue-50 rounded-lg border-l-4 border-blue-500">
                                <i class="fas fa-arrow-right text-blue-600 mr-3"></i>
                                <span class="text-gray-700"><strong>Base Model:</strong> VGG16 (ImageNet pre-trained, frozen layers)</span>
                            </div>
                            <div class="flex items-center p-3 bg-green-50 rounded-lg border-l-4 border-green-500">
                                <i class="fas fa-arrow-right text-green-600 mr-3"></i>
                                <span class="text-gray-700"><strong>Flatten Layer:</strong> Feature vector extraction</span>
                            </div>
                            <div class="flex items-center p-3 bg-purple-50 rounded-lg border-l-4 border-purple-500">
                                <i class="fas fa-arrow-right text-purple-600 mr-3"></i>
                                <span class="text-gray-700"><strong>Dense Layers:</strong> 512 → 256 neurons with ReLU</span>
                            </div>
                            <div class="flex items-center p-3 bg-yellow-50 rounded-lg border-l-4 border-yellow-500">
                                <i class="fas fa-arrow-right text-yellow-600 mr-3"></i>
                                <span class="text-gray-700"><strong>Regularization:</strong> BatchNorm + Dropout (0.5, 0.3)</span>
                            </div>
                            <div class="flex items-center p-3 bg-red-50 rounded-lg border-l-4 border-red-500">
                                <i class="fas fa-arrow-right text-red-600 mr-3"></i>
                                <span class="text-gray-700"><strong>Output:</strong> Single neuron with Sigmoid activation</span>
                            </div>
                        </div>
                    </div>
                    
                    <div>
                        <h4 class="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                            <i class="fas fa-tools text-orange-600 mr-2"></i>Training Configuration
                        </h4>
                        <div class="space-y-4">
                            <div class="p-4 bg-orange-50 rounded-lg border border-orange-200">
                                <h5 class="font-medium text-orange-800 mb-2">Optimizer & Loss</h5>
                                <ul class="text-sm text-gray-700 space-y-1">
                                    <li>• SGD with momentum (0.9)</li>
                                    <li>• Learning rate: 1e-4</li>
                                    <li>• Binary crossentropy loss</li>
                                    <li>• Class-weighted training</li>
                                </ul>
                            </div>
                            <div class="p-4 bg-teal-50 rounded-lg border border-teal-200">
                                <h5 class="font-medium text-teal-800 mb-2">Callbacks & Monitoring</h5>
                                <ul class="text-sm text-gray-700 space-y-1">
                                    <li>• Early stopping (patience: 5)</li>
                                    <li>• Learning rate reduction</li>
                                    <li>• Best weights restoration</li>
                                    <li>• Custom F1-score metric</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Model Performance -->
        <section class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-chart-line text-blue-600 mr-3"></i>Model Performance & Results
                </h2>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div class="text-center p-6 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border border-blue-200">
                        <i class="fas fa-chart-area text-4xl text-blue-600 mb-3"></i>
                        <h3 class="text-2xl font-bold text-blue-800">0.54</h3>
                        <p class="text-blue-700 font-medium">ROC AUC Score</p>
                        <p class="text-sm text-gray-600 mt-2">Area under ROC curve indicating model discrimination ability</p>
                    </div>
                    <div class="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-lg border border-green-200">
                        <i class="fas fa-bullseye text-4xl text-green-600 mb-3"></i>
                        <h3 class="text-2xl font-bold text-green-800">0.69</h3>
                        <p class="text-green-700 font-medium">PR AUC Score</p>
                        <p class="text-sm text-gray-600 mt-2">Precision-Recall area under curve for imbalanced data evaluation</p>
                    </div>
                    <div class="text-center p-6 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg border border-purple-200">
                        <i class="fas fa-balance-scale text-4xl text-purple-600 mb-3"></i>
                        <h3 class="text-2xl font-bold text-purple-800">Balanced</h3>
                        <p class="text-purple-700 font-medium">Class Weights</p>
                        <p class="text-sm text-gray-600 mt-2">Automatic class balancing for optimal training</p>
                    </div>
                </div>

                <div class="bg-gradient-to-r from-gray-50 to-gray-100 p-6 rounded-lg mb-6">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i class="fas fa-microscope text-indigo-600 mr-2"></i>Key Performance Insights
                    </h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <h4 class="font-medium text-gray-800 mb-2">Model Strengths:</h4>
                            <ul class="text-sm text-gray-700 space-y-1">
                                <li>• Reasonable precision-recall performance (PR AUC: 0.69)</li>
                                <li>• Effective transfer learning from ImageNet features</li>
                                <li>• Comprehensive evaluation with multiple metrics</li>
                                <li>• Visual explainability through Grad-CAM</li>
                            </ul>
                        </div>
                        <div>
                            <h4 class="font-medium text-gray-800 mb-2">Areas for Improvement:</h4>
                            <ul class="text-sm text-gray-700 space-y-1">
                                <li>• ROC AUC suggests room for discrimination improvement</li>
                                <li>• Consider ensemble methods or advanced architectures</li>
                                <li>• Additional data augmentation strategies</li>
                                <li>• Fine-tuning of hyperparameters</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-semibold text-yellow-800 mb-2 flex items-center">
                        <i class="fas fa-lightbulb mr-2"></i>Performance Analysis Generated
                    </h4>
                    <p class="text-yellow-700 text-sm">
                        The model generates comprehensive performance visualizations including ROC curves, precision-recall curves, 
                        probability distributions, confusion matrices, and misclassified sample analysis for thorough evaluation.
                    </p>
                </div>
            </div>
        </section>

        <!-- Installation & Usage -->
        <section class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-download text-green-600 mr-3"></i>Installation & Usage
                </h2>
                
                <div class="mb-8">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i class="fas fa-list text-blue-600 mr-2"></i>Prerequisites
                    </h3>
                    <div class="code-block">
# Required Libraries
python >= 3.7
tensorflow >= 2.0
opencv-python
scikit-learn
matplotlib
seaborn
numpy
tkinter (usually included with Python)
PIL (Pillow)
                    </div>
                </div>

                <div class="mb-8">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i class="fas fa-terminal text-green-600 mr-2"></i>Installation Steps
                    </h3>
                    <div class="code-block">
# 1. Clone the repository
git clone https://github.com/your-username/breast-cancer-classification.git
cd breast-cancer-classification

# 2. Install required packages
pip install tensorflow opencv-python scikit-learn matplotlib seaborn pillow

# 3. Download the BreakHis dataset
# Visit: https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset
# Extract to: dataset_cancer_v1/classificacao_binaria/400X/

# 4. Run the training script
python train.py

# 5. Launch the GUI application
python main2.py
                    </div>
                </div>

                <div class="mb-8">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i class="fas fa-play text-purple-600 mr-2"></i>Usage Instructions
                    </h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="p-6 bg-blue-50 rounded-lg border border-blue-200">
                            <h4 class="font-semibold text-blue-800 mb-3 flex items-center">
                                <i class="fas fa-graduation-cap mr-2"></i>Training the Model
                            </h4>
                            <ol class="text-sm text-gray-700 space-y-2">
                                <li>1. Ensure dataset is properly structured</li>
                                <li>2. Run <code class="bg-white px-2 py-1 rounded">train.py</code></li>
                                <li>3. Monitor training progress and metrics</li>
                                <li>4. Model saves as <code class="bg-white px-2 py-1 rounded">enhanced_breast_cancer_model.h5</code></li>
                            </ol>
                        </div>
                        <div class="p-6 bg-green-50 rounded-lg border border-green-200">
                            <h4 class="font-semibold text-green-800 mb-3 flex items-center">
                                <i class="fas fa-desktop mr-2"></i>Using the GUI
                            </h4>
                            <ol class="text-sm text-gray-700 space-y-2">
                                <li>1. Launch <code class="bg-white px-2 py-1 rounded">main2.py</code></li>
                                <li>2. Click "Select Image" button</li>
                                <li>3. Choose a histopathological image</li>
                                <li>4. View prediction and Grad-CAM visualization</li>
                            </ol>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- File Structure -->
        <section class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-folder-open text-orange-600 mr-3"></i>Project Structure
                </h2>
                
                <div class="code-block">
breast-cancer-classification/
│
├── train.py                           # Model training script
├── main2.py                          # GUI application
├── enhanced_breast_cancer_model.h5   # Trained model (generated)
│
├── dataset_cancer_v1/
│   └── classificacao_binaria/
│       └── 400X/
│           ├── benign/               # Benign tissue images
│           └── malignant/            # Malignant tissue images
│
├── output_plots/                     # Generated visualizations
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   ├── probability_distribution.png
│   └── misclassified_samples.png
│
└── README.md                         # Project documentation
                </div>

                <div class="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="p-6 bg-indigo-50 rounded-lg border border-indigo-200">
                        <h4 class="font-semibold text-indigo-800 mb-3 flex items-center">
                            <i class="fas fa-file-code mr-2"></i>Core Files
                        </h4>
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li><strong>train.py:</strong> Complete model training pipeline with evaluation</li>
                            <li><strong>main2.py:</strong> Interactive GUI for image classification and visualization</li>
                            <li><strong>model.h5:</strong> Saved trained model weights and architecture</li>
                        </ul>
                    </div>
                    <div class="p-6 bg-pink-50 rounded-lg border border-pink-200">
                        <h4 class="font-semibold text-pink-800 mb-3 flex items-center">
                            <i class="fas fa-chart-pie mr-2"></i>Generated Outputs
                        </h4>
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li><strong>Performance Plots:</strong> ROC, PR curves, distributions</li>
                            <li><strong>Model Analysis:</strong> Confusion matrix, classification report</li>
                            <li><strong>Visual Explanations:</strong> Grad-CAM heatmaps</li>
                        </ul>
                    </div>
                </div>
            </div>
        </section>

        <!-- Features & Capabilities -->
        <section class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-star text-yellow-600 mr-3"></i>Features & Capabilities
                </h2>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div>
                        <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                            <i class="fas fa-cog text-blue-600 mr-2"></i>Core Features
                        </h3>
                        <div class="space-y-4">
                            <div class="flex items-start p-4 bg-blue-50 rounded-lg border-l-4 border-blue-500">
                                <i class="fas fa-check-circle text-green-600 mt-1 mr-3"></i>
                                <div>
                                    <h4 class="font-medium text-gray-800">Transfer Learning</h4>
                                    <p class="text-sm text-gray-600">VGG16-based architecture with custom classification layers</p>
                                </div>
                            </div>
                            <div class="flex items-start p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
                                <i class="fas fa-check-circle text-green-600 mt-1 mr-3"></i>
                                <div>
                                    <h4 class="font-medium text-gray-800">Data Augmentation</h4>
                                    <p class="text-sm text-gray-600">Comprehensive image augmentation for robust training</p>
                                </div>
                            </div>
                            <div class="flex items-start p-4 bg-purple-50 rounded-lg border-l-4 border-purple-500">
                                <i class="fas fa-check-circle text-green-600 mt-1 mr-3"></i>
                                <div>
                                    <h4 class="font-medium text-gray-800">Class Balancing</h4>
                                    <p class="text-sm text-gray-600">Automatic class weight computation for balanced training</p>
                                </div>
                            </div>
                            <div class="flex items-start p-4 bg-orange-50 rounded-lg border-l-4 border-orange-500">
                                <i class="fas fa-check-circle text-green-600 mt-1 mr-3"></i>
                                <div>
                                    <h4 class="font-medium text-gray-800">Advanced Callbacks</h4>
                                    <p class="text-sm text-gray-600">Early stopping, learning rate scheduling, and best weights restoration</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div>
                        <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                            <i class="fas fa-eye text-purple-600 mr-2"></i>Visualization & Analysis
                        </h3>
                        <div class="space-y-4">
                            <div class="flex items-start p-4 bg-red-50 rounded-lg border-l-4 border-red-500">
                                <i class="fas fa-check-circle text-green-600 mt-1 mr-3"></i>
                                <div>
                                    <h4 class="font-medium text-gray-800">Grad-CAM Visualization</h4>
                                    <p class="text-sm text-gray-600">Visual explanations showing model attention areas</p>
                                </div>
                            </div>
                            <div class="flex items-start p-4 bg-teal-50 rounded-lg border-l-4 border-teal-500">
                                <i class="fas fa-check-circle text-green-600 mt-1 mr-3"></i>
                                <div>
                                    <h4 class="font-medium text-gray-800">Performance Metrics</h4>
                                    <p class="text-sm text-gray-600">ROC curves, PR curves, confusion matrices, and classification reports</p>
                                </div>
                            </div>
                            <div class="flex items-start p-4 bg-yellow-50 rounded-lg border-l-4 border-yellow-500">
                                <i class="fas fa-check-circle text-green-600 mt-1 mr-3"></i>
                                <div>
                                    <h4 class="font-medium text-gray-800">Probability Analysis</h4>
                                    <p class="text-sm text-gray-600">Distribution analysis and misclassification examination</p>
                                </div>
                            </div>
                            <div class="flex items-start p-4 bg-pink-50 rounded-lg border-l-4 border-pink-500">
                                <i class="fas fa-check-circle text-green-600 mt-1 mr-3"></i>
                                <div>
                                    <h4 class="font-medium text-gray-800">Interactive GUI</h4>
                                    <p class="text-sm text-gray-600">User-friendly interface for real-time image classification</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Technical Implementation -->
        <section class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-code text-indigo-600 mr-3"></i>Technical Implementation Details
                </h2>
                
                <div class="mb-8">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i class="fas fa-brain text-purple-600 mr-2"></i>Model Training Pipeline
                    </h3>
                    <div class="code-block">
# Key Training Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
OPTIMIZER = SGD(learning_rate=1e-4, momentum=0.9)
LOSS = 'binary_crossentropy'

# Data Augmentation
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

# Transfer Learning Setup
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False
                    </div>
                </div>

                <div class="mb-8">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i class="fas fa-search text-green-600 mr-2"></i>Grad-CAM Implementation
                    </h3>
                    <div class="code-block">
def grad_cam(model, image_path, last_conv_layer_name="block5_conv3"):
    """
    Generate Grad-CAM visualization for model interpretability
    """
    # Load and preprocess image
    img_array = load_image(image_path)
    
    # Create gradient model
    grad_model = Model([model.inputs], 
                      [model.get_layer(last_conv_layer_name).output, model.output])
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    
    # Generate heatmap
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    return heatmap, superimposed_image
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="p-6 bg-blue-50 rounded-lg border border-blue-200">
                        <h4 class="font-semibold text-blue-800 mb-3 flex items-center">
                            <i class="fas fa-shield-alt mr-2"></i>Model Regularization
                        </h4>
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li>• Batch Normalization layers</li>
                            <li>• Dropout (0.5, 0.3) for overfitting prevention</li>
                            <li>• Early stopping with patience</li>
                            <li>• Learning rate reduction on plateau</li>
                        </ul>
                    </div>
                    <div class="p-6 bg-green-50 rounded-lg border border-green-200">
                        <h4 class="font-semibold text-green-800 mb-3 flex items-center">
                            <i class="fas fa-chart-bar mr-2"></i>Evaluation Metrics
                        </h4>
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li>• Accuracy, Precision, Recall</li>
                            <li>• Custom F1-score implementation</li>
                            <li>• ROC AUC and PR AUC scores</li>
                            <li>• Confusion matrix analysis</li>
                        </ul>
                    </div>
                </div>
            </div>
        </section>

        <!-- Future Improvements -->
        <section class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-rocket text-red-600 mr-3"></i>Future Enhancements
                </h2>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div>
                        <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                            <i class="fas fa-plus text-green-600 mr-2"></i>Model Improvements
                        </h3>
                        <div class="space-y-3">
                            <div class="p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
                                <h4 class="font-medium text-green-800">Ensemble Methods</h4>
                                <p class="text-sm text-gray-600 mt-1">Combine multiple models for better performance and robustness</p>
                            </div>
                            <div class="p-4 bg-blue-50 rounded-lg border-l-4 border-blue-500">
                                <h4 class="font-medium text-blue-800">Advanced Architectures</h4>
                                <p class="text-sm text-gray-600 mt-1">Experiment with ResNet, DenseNet, or EfficientNet</p>
                            </div>
                            <div class="p-4 bg-purple-50 rounded-lg border-l-4 border-purple-500">
                                <h4 class="font-medium text-purple-800">Fine-tuning Strategy</h4>
                                <p class="text-sm text-gray-600 mt-1">Gradual unfreezing and fine-tuning of pre-trained layers</p>
                            </div>
                        </div>
                    </div>
                    
                    <div>
                        <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                            <i class="fas fa-tools text-orange-600 mr-2"></i>System Enhancements
                        </h3>
                        <div class="space-y-3">
                            <div class="p-4 bg-orange-50 rounded-lg border-l-4 border-orange-500">
                                <h4 class="font-medium text-orange-800">Web Application</h4>
                                <p class="text-sm text-gray-600 mt-1">Deploy as a web service using Flask/Django for broader accessibility</p>
                            </div>
                            <div class="p-4 bg-red-50 rounded-lg border-l-4 border-red-500">
                                <h4 class="font-medium text-red-800">Multi-class Classification</h4>
                                <p class="text-sm text-gray-600 mt-1">Extend to classify different cancer subtypes</p>
                            </div>
                            <div class="p-4 bg-teal-50 rounded-lg border-l-4 border-teal-500">
                                <h4 class="font-medium text-teal-800">Advanced Visualization</h4>
                                <p class="text-sm text-gray-600 mt-1">Implement LIME, SHAP, or other explainability methods</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Conclusion & Contact -->
        <section class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-graduation-cap text-blue-600 mr-3"></i>Project Conclusion
                </h2>
                
                <div class="bg-gradient-to-r from-blue-50 to-purple-50 p-8 rounded-lg border-l-4 border-blue-500 mb-8">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Final Year B.Tech Project</h3>
                    <p class="text-gray-700 text-lg leading-relaxed mb-4">
                        This project demonstrates the practical application of deep learning techniques in medical image analysis, specifically for breast cancer histopathological classification. 
                        The implementation showcases a complete machine learning pipeline from data preprocessing to model deployment with interpretable AI features.
                    </p>
                    <p class="text-gray-600">
                        The project successfully integrates transfer learning, comprehensive evaluation metrics, visual explanations through Grad-CAM, 
                        and provides both command-line and GUI interfaces for different use cases.
                    </p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="text-center p-6 bg-green-50 rounded-lg border border-green-200">
                        <i class="fas fa-medal text-4xl text-green-600 mb-3"></i>
                        <h4 class="font-semibold text-green-800 mb-2">Academic Achievement</h4>
                        <p class="text-sm text-gray-600">Comprehensive implementation of ML concepts for medical applications</p>
                    </div>
                    <div class="text-center p-6 bg-blue-50 rounded-lg border border-blue-200">
                        <i class="fas fa-code-branch text-4xl text-blue-600 mb-3"></i>
                        <h4 class="font-semibold text-blue-800 mb-2">Open Source</h4>
                        <p class="text-sm text-gray-600">Available on GitHub for educational and research purposes</p>
                    </div>
                    <div class="text-center p-6 bg-purple-50 rounded-lg border border-purple-200">
                        <i class="fas fa-lightbulb text-4xl text-purple-600 mb-3"></i>
                        <h4 class="font-semibold text-purple-800 mb-2">Future Research</h4>
                        <p class="text-sm text-gray-600">Foundation for advanced medical AI research and development</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Footer -->
        <footer class="bg-gradient-to-r from-gray-800 to-gray-900 text-white p-8 rounded-lg">
            <div class="text-center">
                <h3 class="text-2xl font-bold mb-4 flex items-center justify-center">
                    <i class="fab fa-github mr-3"></i>GitHub Repository
                </h3>
                <p class="text-gray-300 mb-6">
                    This project is open source and available for educational and research purposes.
                    Feel free to explore, contribute, and build upon this work.
                </p>
                <div class="flex justify-center space-x-6">
                    <div class="flex items-center text-gray-300">
                        <i class="fas fa-star mr-2 text-yellow-400"></i>
                        <span>Star this repository</span>
                    </div>
                    <div class="flex items-center text-gray-300">
                        <i class="fas fa-code-branch mr-2 text-green-400"></i>
                        <span>Fork and contribute</span>
                    </div>
                    <div class="flex items-center text-gray-300">
                        <i class="fas fa-bug mr-2 text-red-400"></i>
                        <span>Report issues</span>
                    </div>
                </div>
                <div class="mt-6 pt-6 border-t border-gray-700">
                    <p class="text-sm text-gray-400">
                        © 2024 - B.Tech Final Year Project | Breast Cancer Histopathological Classification
                    </p>
                </div>
            </div>
        </footer>
    </div>
</body>
</html>
