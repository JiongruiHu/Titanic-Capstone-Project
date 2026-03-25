import pandas as pd
import joblib
from pathlib import Path
from preprocessing import data_cleaning

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

def train_logreg(df):

    LogReg_features = [
        "Sex_int","Fare","FamilySize",
        "Pclass_2","Pclass_3",
        "AgeBin_Teen","AgeBin_Adult", "AgeBin_Middle_Aged","AgeBin_Senior",
        "Embarked_C","Embarked_Q",
        "Title_Miss","Title_Mrs","Title_Mr","Title_Rare"
    ]
    missing = [col for col in LogReg_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing LogReg features: {missing}")
   
    X = df[LogReg_features]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Logistic Regression Accuracy: {acc:.4f}")

    # Save model + features together
    joblib.dump({
        "model": model,
        "features": LogReg_features
        }, 
        MODEL_DIR/"titanic_model_LR.pkl")

    return model, acc

def train_rf(df):

    RF_features = [
        "Sex_int","Pclass","Fare","SibSp","Parch",
        "Age","Embarked_int","Title_int","CabinDeck_int",
        ]
    missing = [col for col in RF_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing RF features: {missing}")

    X = df[RF_features]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Random Forest Accuracy: {acc:.4f}")

    # Save model + features together
    joblib.dump(
        {
        "model": model,
        "features": RF_features
        }, 
        MODEL_DIR/ "titanic_model_rf.pkl")

    return model, acc

def main():

    data = pd.read_csv(DATA_PATH)
    processed_data = data_cleaning(data)

    train_logreg(processed_data)
    train_rf(processed_data)

if __name__ == "__main__":
    main()










