# 12433773_fruit-classification-cnn

## Applied Deep Learning

### Image Classification of Fruits Using Convolutional Neural Networks (CNNs)

**Course:** Applied Deep Learning (2025)  
**Student Name:** Mian Azan Farooq  
**Matriculation Number:** 12433773  

---

### 1. Project Overview

The goal of this project is to build an end-to-end deep learning pipeline for image classification using a convolutional neural network. The system classifies fruit images into their respective categories based on the Fruits360 dataset.

The project includes:

- A complete training pipeline
- A validation and testing setup
- A Streamlit-based inference application
- Confidence-based Out-of-Distribution (OOD) detection

This implementation serves as a baseline model, which was iteratively improved and debugged during development.

---

### 2. Dataset

**Dataset:** Fruits360 Dataset

**Source:**(https://www.kaggle.com/moltean/fruits)

- **Training/** – used for training and validation (80/20 split)
- **Test/** – used for evaluation

**Image Size:** 224 × 224

**Color Mode:** RGB (3 channels)

All images are rescaled to [0, 1] during preprocessing.

---

### 3. Error Metric

- **Primary Metric:** Categorical Accuracy
- **Loss Function:** Categorical Crossentropy

**Target Metric:**

- Target accuracy: ≥ 90% validation accuracy

**Achieved Results:**

- Training and validation accuracy exceeded the target for most fruit classes
- High-confidence predictions for in-distribution samples (e.g., apples)
- Lower confidence for ambiguous or out-of-distribution inputs

---

### 4. Model Architecture

**Base model:** EfficientNetB0

**Weights:** Trained from scratch (weights=None)

**Reason:** Avoid shape mismatch and ensure full control over input pipeline

**Architecture:**

- EfficientNetB0 (feature extractor)
- Global Average Pooling
- Dense Softmax classification layer

---

### 5. Training Pipeline

Implemented using **ImageDataGenerator**

- **Validation Split:** 20%
- **Optimizer:** Adam (learning rate = 1e-4)
- **Epochs:** 10

The training pipeline is fully automated and reproducible.

---

### 6. Inference & Streamlit Application

A **Streamlit** web application allows users to upload an image and receive predictions.

**Inference Steps:**

1. Image upload
2. Resize to 224 × 224
3. Normalize pixel values
4. Model prediction
5. Confidence-based decision

---

### 7. Out-of-Distribution (OOD) Detection

To prevent unreliable predictions, an OOD threshold is used:

**OOD_THRESHOLD = 0.60**

Predictions below 60% confidence are flagged as Unknown object (Out-of-Distribution). This prevents overconfident predictions on unrelated images.

**Example:**

- Apple image: 99.61% → accepted
- Non-fruit image (card): 48.36% → low confidence (expected behavior)

This demonstrates proper uncertainty handling.

---

### 8. Testing

- Data loading and preprocessing were manually validated
- End-to-end inference tested using both in-distribution and out-of-distribution images
- Training correctness verified through successful convergence
- Formal unit tests were not implemented due to time constraints, but the pipeline was thoroughly validated during execution.

---

### 9. Software Engineering Practices

- Clear and descriptive variable and function names
- Modular code structure (`train_efficientnet.py`, `data_loader_fruits360.py`, `app_streamlit.py`)
- Comments included where design decisions are non-obvious
- Python version and dependencies documented

---

### 10. Time Breakdown (Approximate)

| Task                              | Time Spent |
|-----------------------------------|------------|
| Dataset setup & exploration       | 3.5 h      |
| Baseline model implementation     | 5 h        |
| Debugging & pipeline fixes       | 5 h        |
| Streamlit app development         | 4 h        |
| Evaluation & testing              | 4.5 h      |
| Documentation                     | 1 h        |

---

### 11. How to Run

**Training:**

```bash
python train_efficientnet.py
```

**Streamlit App:**

```bash
streamlit run app_streamlit.py
```

---

### 12. Conclusion

All requirements of Assignment 2 have been fulfilled:

- Working end-to-end pipeline
- Defined error metric and target
- Achieved and evaluated results
- OOD detection implemented
- Proper software engineering practices followed
