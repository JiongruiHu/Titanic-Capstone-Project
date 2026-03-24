import os
import pickle
import pandas as pd
from django.conf import settings


MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "titanic_model_RF.pkl")

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#MODEL_PATH = os.path.join(BASE_DIR, "titanic_model_RF.pkl")

#print(BASE_DIR)
print(MODEL_PATH)

# Load the model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Convert input data from prediction form into model data form

def preprocess_data(input_data):

    # Convert age input to One Hot encoded AgeBin Features
    age = input_data["age"]
    if age <=12:
        age_bin = "Child"
    elif 12< age <= 19:
        age_bin = "Teen"
    elif 19< age <= 39:
        age_bin = "Adult"
    elif 39< age <= 59:
        age_bin = "Middle Aged"
    else:
        age_bin = "Senior"

    age_bins = ["Teen","Adult","Middle Aged", "Senior"] #Child is the baseline
    agebin_dict = {f"AgeBin_{a}": int(age_bin == a) for a in age_bins}
    
    # Convert embarked input to One Hot encoded Embarked Features
    embarked_bins = ["Embarked_C", "Embarked_Q"] #"S" is the baseline
    embarked_dict = {e: int(input_data["embark"] == e) for e in embarked_bins}
    
    #Convert Title input to One Hot encoded Title Features
    title_bin = ["Miss","Mrs","Mr","Rare"] # Master is the baseline
    title_dict = {f"Title_{t}": int(input_data["title"] == t) for t in title_bin}

    #Convert Pclass input to One Hot encoded Pclass Features
    pclass_bin = [2, 3] # 1 is the baseline
    pclass_dict = {f"Pclass_{t}": int(input_data["passenger_class"] == t) for t in pclass_bin}
    
    '''
    Order of the features in our model as following:
        "Sex","Fare","FamilySize",
        "Pclass_2", "Pclass_3",
        "AgeBin_Child","AgeBin_Teen","AgeBin_Adult", "AgeBin_Middle Aged","AgeBin_Senior",
        "Embarked_C","Embarked_Q","Embarked_S",
        "Title_Miss","Title_Mrs","Title_Mr","Title_Rare",
    '''

    data = {
        "Sex": input_data["gender"],
        "Fare": input_data["ticket_fare"],
        "FamilySize": input_data["siblings_or_spouses"] + input_data["parch"] + 1,
        **pclass_dict,
        **agebin_dict,
        **embarked_dict,
        **title_dict,
              
    }
    print(data)
    # The dataset is Dataframe in our training model, so important to convert to df form
    df = pd.DataFrame([data])
    return df 


def prediction(input_data):
    X = preprocess_data(input_data)
    print(X)  
    prediction_result = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return prediction_result, probability
  