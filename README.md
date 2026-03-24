# Titanic-Capstone-Project

Python AI / ML student project. Capstone project built on machine learning system based on real-world data and deployed as a Django web application.
## Contribution
Forked and extended a collaborative project. My contributions include:
- Data preprocessing
- Feature engineering
- Model building, training and evaluation
## Contents

1. [Project Overview](#project-overview)
2. [How to Set Up files and Run Locally](#how-to-set-up-and-run-locally)
4. [Description of the ML model](#description-of-the-ml-models)
5. [Overview of the System Architecture](#overview-of-the-system-architecture)

## Project Overview

This is a Lexicon course project that demonstrates an end-to-end machine learning and web development workflow. 

It uses historical data from the Titanic disaster to build a model that predicts passenger survival based on features like age, gender, class, and family size. The model is deployed in a web application built with Django, where users can input data and receive predictions with probabilities. The system also stores results in a database to display prediction history. The project emphasizes practical skills such as data preprocessing, model deployment, modular design, and version control using GitHub, simulating a real-world machine learning product lifecycle.

Go back to [Contents](#contents).

## How to Set Up and Run Locally

### Prerequisites

- Python 3.11
- Conda
- pip
- Git

### Clone the repository

```bash
git clone https://github.com/JiongruiHu/Titanic-Capstone-Project.git
cd Titanic-Capstone-Project
```

### Create and activate the virtual environment

```bash
conda env create -f environment.yml
conda activate titanic-capstone-env # or the name specified in environment.yml
```

### Set up the database (first-time setup)

This project uses Django and a SQLite database to store previous predictions.

```bash
python web/manage.py migrate
```

**If you encounter migration issues, ensure the virtual environment is activated before running Django commands.**

### Run the server locally

```bash
python web/manage.py runserver
```

Then open your browser and go to:
http://127.0.0.1:8000/

### Dataset source

The required dataset is already included in the project.

The original dataset source can be found on Kaggle:
https://www.kaggle.com/c/titanic/data

Go back to [Contents](#contents).

## Description of the ML Models

This project solves a binary classification problem: predicting `Survived` (0/1) from passenger features in the Titanic dataset.

Models evaluated:
- Logistic Regression (baseline linear classifier)
- Random Forest (non-linear ensemble baseline)

Feature pipeline (used during training and prediction):
- Missing value handling (Age, Embarked)
- Engineered features: `Title`, `FamilySize`, `AgeBin`, `CabinDeck`
- Categorical encoding for model-ready inputs

Evaluation approach:
- 80/20 train-test split with stratification
- 5-fold stratified cross-validation
- Metrics: Accuracy, Confusion Matrix, Precision/Recall/F1

Result summary:
- Both models achieve about 81% accuracy.
- Logistic Regression provides better recall for survivors and is used for deployment.
- Trained model artifact: `web/predictor/titanic_model.pkl`.

See detailed discussion in [Overview of the System Architecture](#overview-of-the-system-architecture).

Go back to [Contents](#contents).

## Overview of the System Architecture

### 1. Data Preprocessing

#### 1.1 Handling Missing Values

#### Age

Instead of using a global average, missing **Age** values were imputed using **title-based grouping**:

- **Title Extraction:** Extracted from the `Name` column.

  Example:
  - `"Braund, Mr. Owen Harris" → Mr`
  - `"Cumings, Mrs. John Bradley" → Mrs`

- **Title Grouping:**

  A new feature called **Title** by extracting it from **Name** column and grouping titles into **Mr, Mrs, Miss, Master** and **Rare** categories. Missing **Age** values(NaN) are then replaced with the mean age within each title group.

- **Age Imputation:**

  Missing **Age** values were filled with the **mean age** of the corresponding title group.

#### Embarked

This feature contains only two missing values, which are unlikely to affect the model's performance. Therefore, these values are imputed with letter **C**

#### 1.2 Outlier Detection**
  Outlier analysis was performed on the Fare and Age features to identify extreme values.

**Fare**: The IQR (Interquartile Range) method was used because Fare is highly skewed. Although some high fare values were identified as outliers, they were retained since they represent real differences in passenger class and are important for survival prediction.

**Age**: The Z-score method and distribution visualization were used because **Age** is approximately normally distributed. The detected values were within realistic human limits, so no outliers were removed.

Conclusion: No outliers were removed, as they represent valid data and help preserve important patterns for the machine learning model.

#### 1.3 Feature Engineering

- **FamilySize:**
  A new feature called `FamilySize` is created by adding `SibSp + Parch +1`. This represents the **total number of family members aboard**, including the passenger themselves.

- **CabinDeck:**
  `CabinDeck` is extracted from cabin numbers, yielding values `['Unknown', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T']`. 

- **AgeBin:**
  The age feature is grouped into categorical age ranges to reduce noise and capture life-stage patterns that may influence survival. The bins are defined as:
  - **Child:** 0-12 years
  - **Teen:** 13-19 years
  - **Adult:** 20-39 years
  - **Middle Aged:** 40-59 years
  - **Senior:** 60+ years

#### 1.4 Encoding Categorical Variables

One-hot encoding with baseline was used for `AgeBin`,`Embarked`,`Pclass` and `Title` features to prevent the model from assuming any ordinal relationship.

- **Sex:**
  - `male → 0`
  - `female → 1`
- **Embarked(C/Q/S):**
  - `Baseline: Embarked = S`
  - `Embarked_C: 0/1`
  - `Embarked_Q: 0/1`
- **Pclass:**
  - `Baseline: Pclass = 1`
  - `Pclass_2: 0/1`
  - `Pclass_3: 0/1`
- **AgeBin:**
  - `Baseline: AgeBin = Child`
  - `AgeBin_Teen: 0/1`
  - `AgeBin_Adult: 0/1`
  - `AgeBin_Middle Aged: 0/1`
  - `AgeBin_Senior: 0/1`
- **Title:**
  - Grouping titles into **Mr, Mrs, Miss, Master** and **Rare** which were also encoded:
    - `Baseline: Master`
    - `Title_Miss: 0/1`
    - `Title_Mrs: 0/1`
    - `Title_Mr: 0/1`
    - `Title_Rare: 0/1`
  - Each title category is converted into a binary (0/1) column.This allows logistic regression to learn survival patterns related to:
    - Gender (Mr vs Mrs vs Miss)
    - Age group (Master = young boys)
    - Social status (Rare titles)

- **Example:**

  Table 1: Original Title Feature Before One-Hot Encoding with `Master` as baseline
  | PassengerId | Title |
  | ----------- | ------ |
  | 1 | Mr |
  | 2 | Mrs |

  Table 2: Title Features After One-Hot Encoding
  | PassengerId | Title_Miss | Title_Mrs | Title_Mr | Title_Rare |
  | ----------- | ---------- | --------- | -------- | ---------- |
  | 1 | 0 | 0 | 1 | 0 |
  | 2 | 0 | 1 | 0 | 0 |

### 2. Model Training and Evaluation

This section describes how the machine learning models were selected, trained, validated, and evaluated for Titanic survival prediction. It summarizes the modeling decisions, compares Logistic Regression and Random Forest performance, and explains why the final deployed model was chosen.

#### 2.1 Model Selection (why Logistic Regression or Random Forest) 

- **Problem Definition**

  The aim is to predict whether a passenger survived the Titanic disaster. This is a binary classification task using the Titanic dataset, which contains passenger information such as age, sex, passenger class, and other relevant features.

- **Baseline Model: Logistic Regression (LR)**  
  Logistic Regression is chosen as the baseline because it is a simple and widely used model for binary classification tasks. LR assumes a linear relationship between input features and log-odds of the target outcomes, providing a clear and interpretable reference point for comparing more complex models.

  **Prediction Pipeline**
    1. **Feature Preparation:**
        - Engineer additional features:
          - **AgeBin:** Convert `Age` into categorical age groups (e.g., Child, Adult, Senior)
          - **Title:** Extract titles from passenger names (e.g., Mr, Mrs, Miss, Master, Rare) and encode as categorical
          - **FamilySize:** Compute total family size `SibSp` + `parch` + 1
  
        - Encode categorical features `Sex`,`AgeBin`, `Embarked`, `Title`,`Pclass` using one-hot encoding with baseline.
        - Keep numerical features `Fare`, `FamilySize` as is.
    2. **Probability Computation:**
        - Linear Weighted Sum of the Features:
        $$z = \beta_0 + \beta_1 \text{Sex} + \beta_2 \text{Pclass} + \beta_3 \text{Fare} + \beta_4 \text{AgeBin} + \beta_5 \text{Title} + \dots$$
        - The Predicted Probability of Survival:
  
            $$P(\text{Survived}=1) = \frac{1}{1 + e^ {- z}}$$, 
   assign class based on probability threshold (commonly 0.5): - P>0.5⇒Survived - P≤0.5⇒Did not survive.

- **Candidate Model: Random Forest (RF)**  

  Random Forest is a supervised ensemble ML method that uses many decision trees trained on random subsets of data and features. Each tree makes a prediction, and the final output is determined by majority voting. Using this model allows us to compare a nonlinear, tree-based approach to feature handling and
  prediction with the linear, weight-based approach used in Logistic Regression.

  **Prediction Pipeline**
  1. **Feature Preparation:**
     - Engineer additional features:
       - **AgeBin:** Convert `Age` into categorical age groups (e.g., Child, Adult, Senior)
       - **Title:** Extract titles from passenger names (e.g., Mr, Mrs, Miss, Master, Rare) and encode as categorical
       - **CabinDeck:** Extract letters from `Cabin` into categorical groups (e.g., `['Unknown', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T']`).
     - Encode categorical features `AgeBin`, `Embarked`, `Title`,`Pclass` and `CabinDeck` using one-hot encoding with baseline, and encode `Sex` as 0 or 1.
     - Keep numerical features `Fare`, `SibSb`, `parch` as is.

  3. **Prediction**
     
  A Random Forest predicts survival by building many decision trees using features like age, gender, and passenger class. Each tree makes its own prediction, and the model combines them through majority voting to decide whether a passenger survived. This ensemble approach improves accuracy and reduces overfitting compared to a single decision tree

- **Logistic Regression vs Random Forest Pipeline Comparison**

| Aspect                   | Logistic Regression                                                    | Random Forest                                      |
|--------------------------|------------------------------------------------------------------------|----------------------------------------------------|
| **Categorical Features** | Must be one-hot encoded; baseline category dropped (`drop_first=True`) | Can be integer-mapped or one-hot encoded           |
| **Prediction**           | Linear weighted sum passed through Sigmoid                             | Majority vote across all trees                     |
| **Strength**             | Low computational cost compared to ensemble models                     | Captures non-linear relationships and interactions |
| **Limitation**           | Cannot automatically capture interactions or non-linear effects        | Predictions based on majority vote of trees        |
| **Pipeline**             | Requires scaling and encoding in a structured preprocessing pipeline   | Works with minimal preprocessing; handles features more flexibly |

  
#### 2.2 Validation Strategy

- **Train, Test and Validation Datasets**

  The Titanic dataset has 891 rows, which is relatively small. A 0.1 test split gives more training data, but the test set would only have 89 rows, making the metrics less stable. A 0.3 test split provides a larger test set but reduces the training data, which could slightly hurt model performance.

  Therefore, We split data with `test_size=0.2` (80% train / 20% test), balancing enough training data with a sufficiently large test set for stable evaluation.
  - Training dataset = 713 rows → enough to train logistic regression
  - Test dataset = 178 rows →enough to get stable f1 scores

  We use **Stratified K-Fold Cross-Validation** :

  The Titanic dataset has 891 rows, which is relatively small. A 0.1 test split gives more training data, but the test set would only have 89 rows, making the metrics less stable. A 0.3 test split provides a larger test set but reduces the training data, which could slightly hurt model performance.

  Therefore, We split data with `test_size=0.2` (80% train / 20% test), balancing enough training data with a sufficiently large test set for stable evaluation.
  - Training dataset = 713 rows → enough to train logistic regression
  - Test dataset = 178 rows →enough to get stable f1 scores
    Split training data into k folds, train on k-1 folds and validate on the remaining fold and repeat k times. The class distribution in each fold is fixed

  ```python
  cv = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
  f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
  ```

#### 2.3 Evaluation Metrics

The Titanic Dataset is slightly imbalanced as shown below:

| Class            | Count | Percentage |
| ---------------- | ----- | ---------- |
| 0 (Not survived) | 549   | ~62%       |
| 1 (Survived)     | 342   | ~38%       |

With 62% non-survivors (Majority Class) and 38% survivors (Minority Class). The accuracy is biased towards to non-survivors, therefore we focus on `f1` for the survivors (minority class).
And use `stratify=y` in `train_test_split` so that **the class distribution in train and test sets matches the original distribution of y**.

- **Logistic Regression Model Overall Performance**

  The model performs well at distinguishing between survivors and non-survivors, correctly identifying most passengers. It occasionally overestimates survival, but overall the confusion matrix shows that the model makes relatively few misclassifications and captures the patterns in the data effectively.

  ![confusion matrix](web/static/images/confusion_matrix.jpg)

  | Class               | Precision | Recall | F1-Score  | Support |
  | ------------------- | --------- | ------ | --------- | ------- |
  | **0 – Not Survive** | ✔0.87     | ✔0.80  | ✔0.83     | 105     |
  | **1 – Survived**    | •0.74     | ✔0.82  | •**0.78** | 74      |
  | **Accuracy**        |           |        | ✔**0.81** | 179     |
  | **Macro Avg**       | 0.80      | 0.81   | 0.81      | 179     |
  | **Weighted Avg**    | 0.82      | 0.81   | 0.81      | 179     |

  Cross validation gives Mean F1-score = 0.7566 which indicates that our Logistic Regression model performs generally well too.

- **Random Forest Model Overall Performance**

  Cross validation gives Mean F1-score = 0.7660 which indicates that our RF model performs generally well.

  | Class                | Precision | Recall    | F1-score  | Support |
  | -------------------- | --------- | --------- | --------- | ------- |
  | **0 – Not Survived** | ✔0.82     | ✔0.88     | ✔0.85     | 110     |
  | **1 – Survived**     | ✔0.79     | •**0.70** | •**0.74** | 69      |
  | **Accuracy**         |           |           | **0.81**  | 179     |
  | **Macro Avg**        | 0.80      | 0.79      | 0.79      | 179     |
  | **Weighted Avg**     | 0.81      | 0.81      | 0.81      | 179     |

- **Comparison of prediction performance: LR vs RF**
  - **Accuracy** : both models give high accuracy

    | Model               | Accuracy |
    | ------------------- | -------- |
    | Random Forest       | ~81%     |
    | Logistic Regression | ~81%     |

  - **Recall on survivors** : LR is **much better at detecting survivors**, and RF misses more survivors.

    | Model | Recall (Survived) |
    | ----- | ----------------- |
    | RF    | 0.70              |
    | LR    | 0.82              |

  - **Precision on survivors** : RF is slightly more precise at prediction on survivor.

    | Model | Precision (Survived) |
    | ----- | -------------------- |
    | RF    | 0.79                 |
    | LR    | 0.74                 |

Both Logistic Regression and Random Forest achieved similar overall accuracy (~81%). However, Logistic Regression is more balanced and Random Forest is slightly biased toward predicting non-survival.

### 3. Django Integration

The trained machine learning model is deployed inside the Django web application to provide real-time survival predictions. The integration consists of model loading, form handling and validation, preprocessing of user input, prediction generation, and displaying results.


Go back to [Contents](#contents).
