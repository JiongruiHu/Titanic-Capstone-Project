import pandas as pd

def data_cleaning(raw_data):
    df = raw_data.copy()
  
    # -----------------------
    # Basic cleaning
    # -----------------------

    # Title extraction
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.',expand=False)
    
    df['Title'] = df['Title'].where(
        df['Title'].isin(['Mr', 'Mrs', 'Master', 'Miss']),
        'Rare'
    )

    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Fill Age
    df['Age'] = df['Age'].fillna(
        df.groupby('Title')['Age'].transform('mean')
    )

    # Fill Embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = pd.Categorical(
        df['Embarked'],
        categories=['S', 'C', 'Q'],  #  make sure 'S' is the baseline
        ordered=True
        )
    # -----------------------
    # Age binning
    # -----------------------

    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 45, 60, float("inf")],
        labels=['Child','Teen','Adult','Middle_Aged','Senior']
    )
    # -----------------------
    # CabinDeck Extraction
    # -----------------------
    df["CabinDeck"] = (
        df["Cabin"]
        .str.extract(r"([A-Za-z])", expand=False)
        .str.upper()
        .fillna("Unknown")
    )

    # -----------------------
    # Encoding features
    # -----------------------

    # One-hot encoding with baseline (for LogReg)
    df['Pclass'] = df['Pclass'].astype('category')
    cols_to_encode = ['Pclass', 'Embarked', 'Title', 'AgeBin']
    df_dummies = pd.get_dummies(df[cols_to_encode], dtype=int, drop_first=True)

    df_encoded = pd.concat([df, df_dummies], axis=1)

    # Integer encoding 
    title_map = {'Mr':0, 'Mrs':1, 'Master':2, 'Miss':3, 'Rare':4}
    embarked_map = {'S':0, 'C':1, 'Q':2}
    cabin_map = {
        'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6,'T':7, 'Unknown':8
        }
    sex_map ={'male':0, 'female':1}

    df_encoded['Title_int'] = df_encoded['Title'].map(title_map).fillna(-1)
    df_encoded['Embarked_int'] = df_encoded['Embarked'].map(embarked_map).fillna(-1)
    df_encoded['CabinDeck_int'] = df_encoded['CabinDeck'].map(cabin_map).fillna(-1)
    df_encoded['Sex_int'] = df_encoded['Sex'].map(sex_map).fillna(-1)

    
    return df_encoded