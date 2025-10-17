# Plant-Disease-Detection-using-Deep-learning-1
# 🍅 Tomato Leaf Disease Detection using Deep Learning (ResNet50)

## 📘 Overview
This project applies **Transfer Learning** using a pre-trained **ResNet50** CNN to classify tomato leaf diseases into **9 categories**.  
The fine-tuned model achieves **98.7% test accuracy**, enabling early detection of plant diseases to support **smart and sustainable agriculture**.  

This work is part of the **HIT401 Capstone Project** at **Charles Darwin University**.

---

## ⚙️ Environment Configuration

### 🧩 Requirements
```bash
Python 3.9+
TensorFlow / Keras 2.12+
NumPy 1.23+
Matplotlib 3.7+
scikit-learn 1.3+
Pandas 1.5+
##Optional Tools

- **Jupyter Notebook** or **Google Colab** for training and visualization  
- **Git LFS (Large File Storage)** for uploading large datasets  

---

Hardware

- GPU (Recommended):NVIDIA CUDA-enabled device  
- CPU:Works, but training is significantly slower  

---

Parameter Settings

| Parameter | Description | Value |
|------------|-------------|--------|
| `image_size` | Input image dimension | (256, 256, 3) |
| `batch_size` | Number of images per batch | 16 |
| `base_model` | Transfer learning backbone | ResNet50 (ImageNet weights) |
| `optimizer` | Optimization algorithm | Adam |
| `learning_rate` | Initial learning rate | 0.0001 |
| `dropout_rate` | Regularization to prevent overfitting | 0.3 |
| `epochs` | Fine-tuning duration | 3 |
| `loss_function` | Objective | Sparse Categorical Cross-Entropy |
| `metrics` | Evaluation metrics | Accuracy, Precision, Recall, F1-score |
| `augmentation` | Random flip, rotation, zoom | Enabled |

---
Execution Steps
1️. Clone Repository
```bash
git clone https:https://github.com/Aayan-code/Plant-Disease-Detection-using-Deep-learning-1
2. Set Up Environment
pip install -r requirements.txt

3. Add Dataset

Dataset of Tomato Leaves.
If the dataset is large, upload it via Google Drive and link it in this README.

4. Train Model
python train_model.py


or in Jupyter:

!jupyter notebook tomato0.ipynb

5. Fine-Tune ResNet50

Unfreeze the top 30 layers for fine-tuning:

for layer in base_model.layers[-30:]:
    layer.trainable = True

6. Evaluate Model
python evaluate_model.py


This will display:

Test accuracy

Classification report

Confusion matrix

Accuracy & loss curves

7. Save & Export Model
model.save('tomato_model_finetuned.keras')

📊 Results Summary

Test Accuracy: 98.7 %

Validation Accuracy: 98.6 %

Test Loss: 0.04

Macro F1 Score: 0.117

📈 Visual Outputs

Confusion Matrix (Before & After Fine-Tuning)

Training Accuracy and Loss Curves

Correct Prediction Samples (Visual Verification)

