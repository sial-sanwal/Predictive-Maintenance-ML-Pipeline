# Predictive Maintenance – Machine Failure Prediction

This project is a complete end-to-end machine learning solution that predicts whether a machine will fail within the next 24 hours using the Microsoft Azure Predictive Maintenance dataset.

---

## 🔗 Dataset
Source: [Kaggle - Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictivemaintenance/data)

---

## 🚀 Project Structure

├── data/
│ ├── raw/ # Raw input CSV files from Kaggle
│ └── output/ # Prediction output file
├── notebooks/
│   └── EDA.ipynb
├── src/
│ ├── data_loader.py # Loads raw data files
│ ├── preprocess.py # Preprocess and feature engineering
│ ├── model.py # ML pipeline definition
│ ├── train.py # End-to-end training script
│ └── predict.py # Inference on test files
├── test_telemetry.csv # Example test input
├── test_machines.csv # Example test input
├── model.pkl # Trained model artifact
├── requirements.txt # Python dependencies
└── README.md # Project overview and instructions



## 📊 Notebooks

The `notebooks/EDA.ipynb` file contains exploratory data analysis (EDA), including:

- Dataset overview
- Missing value checks
- Visualizations of telemetry variables
- Machine metadata insights
- Failure type distributions

This notebook helped inform the feature engineering and preprocessing steps.


## 🛠️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Train the Model
```bash
python src/train.py
```

### 3. Make Predictions on Test Data

```bash
python src/predict.py
```
Output will be saved to ```data/output/predictions.csv.```


### Modeling Approach

**Aggregation:** 3-hour window telemetry aggregation

**Features:** Telemetry signals, machine model, machine age

**Target:** Binary label for machine failure within 24 hours

**Model:** RandomForestClassifier with balanced class weights

**Pipeline:** StandardScaler + RandomForest wrapped in a sklearn pipeline


### Evaluation Metric

Model performance is reported via ```classification_report```, including:

- Accuracy

- Precision

- Recall

- F1 Score

### Trade-offs
- Focused on end-to-end reproducibility and engineering clarity.
- Not tuned for highest possible accuracy.
- RandomForest chosen for simplicity and interpretability.


