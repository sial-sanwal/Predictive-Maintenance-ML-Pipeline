# Predictive Maintenance – Machine Failure Prediction

This project is a complete end-to-end machine learning solution that predicts whether a machine will fail within the next 24 hours using the Microsoft Azure Predictive Maintenance dataset.

---

## 🔗 Dataset
Source: [Kaggle - Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictivemaintenance/data)

---

## 🚀 Project Structure
```
├── data/
│   ├── raw/               # Raw CSV files from Kaggle
│   └── output/            # Prediction results
├── notebooks/
│   └── EDA.ipynb          # Exploratory Data Analysis
├── src/
│   ├── data_loader.py     # Data loading utilities
│   ├── preprocess.py      # Feature engineering & transformation
│   ├── model.py           # ML model pipeline definition
│   ├── train.py           # Training script
│   └── predict.py         # Prediction script
├── test_telemetry.csv     # Sample test input
├── test_machines.csv      # Sample machine metadata
├── artifacts/
│   └── model.pkl          # Trained model (saved after training)
├── requirements.txt       # Python dependencies
├── Dockerfile             # For containerizing the app
├── app.py                 # FastAPI app for deployment
├── train_pipeline.py      # Production Training Pipeline
└── README.md              # Project documentation
```



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
### 2. ```src/train.py``` — Minimal Training Script (For Quick Tests)
This is a lightweight training script for quick development/testing.

Run with:

```bash
python src/train.py
```
**What it does:**
- Loads raw telemetry, machine, and failure data.

- Merges and preprocesses data.

- Trains the model and saves it to model.pkl.

- Prints classification report to console.


### ```train_pipeline.py```  — Full Production Training Pipeline
This script includes full metrics logging and directory structure, recommended for production.

Run With:
```bash
python train_pipeline.py
```
**What it does:**
- Loads and preprocesses all raw data.

- Splits data into training and testing sets.

- Trains the model using ```build_model()```.

- Saves the model to ```artifacts/model.pkl```.

- Generates and saves classification metrics to ```artifacts/metrics.json.```

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


