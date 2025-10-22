# 🌿 Plant Disease Detection Using Deep Learning  
### 🎓 HIT401 Capstone Project | Charles Darwin University  

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.x-red?logo=keras)
![License](https://img.shields.io/badge/License-Academic-blue)
![GPU](https://img.shields.io/badge/Runtime-GPU%20(A100)-green)

---

## 📘 Project Overview  
This project implements an end-to-end **deep learning model** for **tomato leaf disease detection and classification** using **ResNet50** with transfer learning.  
Two datasets — **PlantVillage** (controlled conditions) and **Taiwan Tomato Leaves** (real-world field images) — were combined to improve model robustness and generalization.

The system aims to support **precision agriculture** by automatically identifying tomato leaf diseases, promoting early diagnosis and yield protection for farmers.

---

## ⚙️ Environment Configuration  

| Component | Version / Setting |
|------------|------------------|
| **Platform** | Google Colab Pro+ (A100 GPU) |
| **Python** | 3.12 |
| **TensorFlow / Keras** | 2.16 / 3.x |
| **CUDA / cuDNN** | CUDA 12+ (Colab default) |
| **Libraries** | NumPy, Matplotlib, Seaborn, scikit-learn, OpenCV, gdown |
| **Hardware Acceleration** | Mixed-Precision Training (Enabled) |
| **Runtime Memory** | 12–15 GB (High-RAM recommended) |

---

## 🧮 Parameter Settings  

| Parameter | Description | Value |
|------------|--------------|-------|
| `IMAGE_SIZE` | Input image dimensions | (256 × 256) |
| `BATCH_SIZE` | Batch size for training | 16 |
| `EPOCHS` | Initial training epochs (frozen base) | 10 |
| `FINE_TUNE_EPOCHS` | Deep fine-tuning epochs | 15 |
| `LEARNING_RATE` | Adam optimizer (base) | 3 × 10⁻⁴ |
| `FINE_TUNE_LR` | Reduced learning rate (adaptive) | 2 × 10⁻⁵ → auto reduce |
| `LOSS` | Sparse Categorical Cross-Entropy | Multi-class |
| `CALLBACKS` | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau | Optimization control |
| `DATA_AUGMENTATION` | Flip, rotate, zoom, contrast, brightness | Training set only |

---

## 🧠 Model Architecture  

- **Base:** ResNet50 pretrained on ImageNet  
- **Top Layers:**  
  `GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(num_classes, Softmax)`  
- **Fine-Tuning:** Gradual unfreezing of deeper layers (30–155)  
- **Regularization:** Dropout + L2 kernel regularization  
- **Optimizer:** Adam with adaptive learning rate scheduling  
- **Training Precision:** Mixed float16 (for faster GPU execution)

---

## 🧩 Execution Workflow  

<details>
<summary>1️⃣ Dataset Preparation</summary>

- Download `tomato_dataset.zip` from Google Drive.  
- Extract into `/Dataset of Tomato Leaves/` directory.  
- Folder structure:


Dataset of Tomato Leaves/
├── plantvillage/
│ └── 5 cross-validation/
└── taiwan/
└── data augmentation/

- Both datasets merged to ensure balanced representation.
</details>

<details>
<summary>2️⃣ Data Loading & Preprocessing</summary>

```python
plant_ds = tf.keras.preprocessing.image_dataset_from_directory(plant_path)
taiwan_ds = tf.keras.preprocessing.image_dataset_from_directory(taiwan_path)
combined_ds = plant_ds.concatenate(taiwan_ds)


Split into 80% Train, 10% Validation, 10% Test

Applied normalization (Rescaling(1./255)) and augmentation

Prefetching enabled for GPU optimization

</details> <details> <summary>3️⃣ Model Training Phases</summary>

Stage 1: Train dense head with ResNet50 frozen

Stage 2: Unfreeze upper 100–155 layers → fine-tune

Stage 3: Enable mixed-precision for performance gains

Stage 4: Apply adaptive callbacks

EarlyStopping: monitors validation accuracy

ReduceLROnPlateau: dynamically lowers LR

ModelCheckpoint: saves best weights

</details> <details> <summary>4️⃣ Evaluation & Visualization</summary>

Test Accuracy: ~81 %

Validation Accuracy: ~85 %

Macro F1-Score: ~0.84

Confusion Matrix plotted via Seaborn

Grad-CAM heatmaps show feature localization over lesions and infected areas.

</details> <details> <summary>5️⃣ Reproducibility & Deployment</summary>

Model saved as best_model.keras and tomato_model_finetuned_fast.keras

Compatible with TensorFlow Lite for mobile apps

Easily re-trainable using same data structure in Colab or local GPU

</details>
📊 Results Summary
Metric	Value
Training Accuracy	94.9 %
Validation Accuracy	85.8 %
Test Accuracy	81.2 %
Test Loss	0.90
F1-Score (Macro)	0.84
Best Model	Mixed-Precision Fine-Tuned ResNet50
🔬 Grad-CAM Insights
Observation	Description
Healthy Leaves	Model focuses on entire surface uniformity
Leaf Mold / Early Blight	Attention on dark necrotic patches
Yellow Leaf Curl Virus	Heatmaps highlight distorted leaf edges
Powdery Mildew	Concentration on white fungal textures
🧭 Future Enhancements

Expand dataset with more field-captured samples

Apply object detection for multi-leaf scenes

Deploy as mobile application for farmers

Explore EfficientNetV2 or Vision Transformers (ViT) for higher accuracy

📁 Repository Structure
Plant-Disease-Detection/
 ├── Dataset of Tomato Leaves        # All Data about tomato leaves 
 ├── gitattributes                   # configure Git LFS for dataset images                   
 ├── README.md                       # Documentation file  
 └── Tomatoo.ipynb                   #Full training and evaluation notebook/Output metrics and Grad-CAM visuals

⚠️ Notes

GPU Runtime Required: Enable GPU in Colab → Runtime → Change runtime type → GPU

Approx. Training Time: ~2 hrs (25 epochs total on A100 GPU)

Dataset Source:
Mendeley Data (DOI: 10.17632/ngdgg79rzb.1
)

Ethical Use: For academic and research purposes only

👨‍💻 Author

HIT401-017 Plant Disease Recognition using Deep Learning
Bachelor of Information Technology
Charles Darwin University – 2025

