# Gait-Based Person Identification System

## Project Overview

This project is a gait-based person identification system that identifies individuals using smartphone accelerometer and gyroscope sensor data.
The system extracts motion-based features from walking patterns and uses machine learning models to classify individuals with high accuracy.

---

## Problem Statement

Traditional authentication methods such as passwords and PINs can be forgotten, guessed, or stolen.
Biometric methods like fingerprint or face recognition require direct user interaction.

This project provides a **behavioral biometric solution** using walking patterns (gait) for identification.

### Project Goals

* Identify individuals based on their walking style
* Use smartphone accelerometer and gyroscope data
* Achieve more than 80% accuracy
* Work with real-world noisy sensor data

---

## Dataset Information

### Training Dataset

Base dataset used in this project:

UCI Human Activity Recognition Using Smartphones Dataset:
https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

Dataset details:

* Contains data of 30 individuals
* Accelerometer (X, Y, Z)
* Gyroscope (X, Y, Z)
* CSV file format

Each person has their own folder:

```
data/raw/person1/
data/raw/person2/
...
```

Each folder contains:

```
acc.csv
gyro.csv
```

---

### Unknown User Data

For prediction, place files here:

```
data/unknown/acc.csv
data/unknown/gyro.csv
```

---

## Project Structure

```
gait_person_identification/
│
├── data/
│   ├── raw/
│   │   ├── person1/
│   │   │   ├── acc.csv
│   │   │   └── gyro.csv
│   │   └── ...
│   │
│   ├── merged/
│   │   └── merged sensor files
│   │
│   ├── processed/
│   │   └── extracted feature files
│   │
│   └── unknown/
│       ├── acc.csv
│       └── gyro.csv
│
├── scripts/
│   ├── merge_sensors.py
│   ├── feature_extraction.py
│   ├── data_augmentation.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── models/
│   └── trained model files
│
├── evaluation/
│   └── metrics and confusion matrix
│
├── notebooks/
│   └── experiment.ipynb
│
├── config.yaml
├── requirements.txt
├── llm_usage.md
└── README.md
```

---

## Installation

### Step 1: Clone the Repository

```
git clone https://github.com/AshishPancheshwar1234/gait-person-identification.git
cd gait_person_identification
```

### Step 2: Create a Virtual Environment (Recommended)

```
python -m venv venv
```

Activate the environment:

**Windows:**

```
venv\Scripts\activate
```

**Mac/Linux:**

```
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required libraries using:

```
pip install -r requirements.txt
```

---

## How to Run the Project (Step-by-Step)

### Step 1: Merge Accelerometer and Gyroscope Data

```
python scripts/merge_sensors.py
```

Output:

```
data/merged/
```

---

### Step 2: Extract Features

```
python scripts/feature_extraction.py
```

Output:

```
data/processed/
```

---

### Step 3: (Optional) Data Augmentation

```
python scripts/data_augmentation.py
```

---

### Step 4: Train the Model

```
python scripts/train_model.py
```

Output:

```
models/
```

---

### Step 5: Evaluate the Model

```
python scripts/evaluate_model.py
```

Output:

```
evaluation/
```

---

### Step 6: Predict Unknown User

1. Place files inside:

```
data/unknown/
```

Required files:

```
acc.csv
gyro.csv
```

2. Run prediction:

```
python scripts/predict.py
```

---

## System Workflow

1. Raw sensor data is collected per person.
2. Accelerometer and gyroscope data are merged.
3. Data is divided into fixed-size windows.
4. Statistical features are extracted from each window.
5. Data augmentation increases training samples.
6. Machine learning model is trained.
7. Model is evaluated.
8. Unknown user is predicted using trained model.

---

## Features Extracted

* Mean
* Standard deviation
* Variance
* RMS (Root Mean Square)
* Signal Magnitude Area

---

## Models Used

* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

The best model is selected based on accuracy.

---

## Results

* Dataset accuracy: ~85–90%
* Real-world accuracy: ~80–85%

### Validation Methods

* Train-test split
* Cross-validation
* Confusion matrix

---

## Author

Ashish Pancheshwar
