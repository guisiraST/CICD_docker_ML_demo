import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import skops.io as sio

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DATA_FILE = "Data/drug.csv"
MODEL_FILE = "Model/drug_pipeline.skops"
METRICS_FILE = "Results/metrics.txt"
RESULTS_IMAGE = "Results/model_results.png"
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_ESTIMATORS = 50

def ensure_directories():
    """Ensure necessary directories exist."""
    os.makedirs("Data", exist_ok=True)
    os.makedirs("Model", exist_ok=True)
    os.makedirs("Results", exist_ok=True)

def load_data(file_path):
    """Load and shuffle the dataset."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df.sample(frac=1, random_state=RANDOM_STATE)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """Split data into train and test sets."""
    if "Drug" not in df.columns:
        raise ValueError("Target column 'Drug' not found in the dataset.")

    X = df.drop("Drug", axis=1).values
    y = df["Drug"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logging.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

def build_pipeline():
    """Build the preprocessing and model pipeline."""
    cat_col = [1, 2, 3]
    num_col = [0, 4]

    transform = ColumnTransformer(
        transformers=[
            ("encoder", OrdinalEncoder(), cat_col),
            ("num_imputer", SimpleImputer(strategy="median"), num_col),
            ("num_scaler", StandardScaler(), num_col),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", transform),
            ("model", RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)),
        ]
    )
    logging.info("Pipeline created successfully.")
    return pipeline

def evaluate_model(pipe, X_test, y_test):
    """Evaluate the model and save results."""
    predictions = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    logging.info(f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")

    with open(METRICS_FILE, "w") as outfile:
        outfile.write(f"Accuracy = {accuracy:.2f}, F1 Score = {f1:.2f}.\n")

    cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
    disp.plot()
    plt.savefig(RESULTS_IMAGE, dpi=120)
    logging.info(f"Evaluation results saved to {METRICS_FILE} and {RESULTS_IMAGE}.")

    return accuracy, f1

def save_model(pipe, model_file):
    """Save the trained pipeline."""
    try:
        sio.dump(pipe, model_file)
        logging.info(f"Model saved successfully to {model_file}.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main():
    ensure_directories()

    # Load and preprocess data
    df = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Build and train the pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    logging.info("Model training complete.")

    # Evaluate the model
    evaluate_model(pipeline, X_test, y_test)

    # Save the trained model
    save_model(pipeline, MODEL_FILE)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Application error: {e}")
