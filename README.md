# Water Body Segmentation and Area Estimation using Deep Learning

This project presents a deep learning-based approach to automatically segment water bodies from satellite or aerial images and estimate their real-world surface area. The system uses a U-Net architecture with a ResNet50 encoder to accurately detect water regions and calculate area using pixel-to-area calibration.

---

## 📌 Project Overview

Water bodies such as lakes, rivers, and reservoirs play a vital role in environmental monitoring, agriculture, and water resource management. Traditional methods for identifying and measuring water bodies are manual, time-consuming, and prone to errors.  
This project provides an automated solution using semantic segmentation and pixel calibration to quickly and accurately estimate water body areas from images.

---

## 🚀 Key Features

- Automatic water body detection using U-Net + ResNet50
- Pixel-to-area calibration using known reference images
- Area estimation in:
  - Square meters (m²)
  - Hectares (ha)
  - Square kilometers (km²)
- Side-by-side visualization of:
  - Original image
  - Predicted water mask
- Works on unseen satellite or aerial images

---

## 🧠 Model Architecture

- **Model:** U-Net with ResNet50 encoder (pretrained on ImageNet)
- **Input Size:** 256 × 256 RGB images
- **Loss Function:** Dice Loss + Binary Cross Entropy
- **Framework:** PyTorch

---

## 🗂️ Dataset

The dataset consists of RGB satellite/aerial images with corresponding binary masks:
- Water regions labeled as **1 (white)**
- Non-water regions labeled as **0 (black)**

All images are resized to 256×256 pixels. Data augmentation is applied using Albumentations to improve model generalization.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- segmentation-models-pytorch
- OpenCV
- NumPy
- Matplotlib
- Albumentations
- Pillow

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Water-Body-Segmentation-Area-Estimation.git
cd Water-Body-Segmentation-Area-Estimation

Step 2: Install Dependencies
      pip install -r requirements.txt

Step 3: Run the Project
      python main.py(for model training)
      python test.py(run the project)