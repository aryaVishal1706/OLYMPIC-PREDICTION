from django.http import HttpResponse
from django.shortcuts import render
from django.core.cache import cache

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
# from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC

# import xgboost as xgb

# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Embedding
# from keras.preprocessing.text import Tokenizer


def about(request):
    return render(request, 'about.html')

def index(request):
    return render(request, 'index.html')

# Perform data preprocessing and generate necessary statistics

df = pd.read_csv("./templates/medals.csv")
df1=df
del df['country_code']
del df['country_3_letter_code']
All_Athlete_URL=set()

for x in df['athlete_url']:
  All_Athlete_URL.add(x)

del df['athlete_url']

df['athlete_full_name']=df.groupby(['discipline_title','medal_type'])['athlete_full_name'].transform(lambda x: ', '.join(str(i) for i in x))
df=df.drop_duplicates(subset=['discipline_title','slug_game','medal_type'])
df['athlete_full_name'].fillna('',inplace=True)

# 4 Knowing the Names of all participants participated till now
def participant(request):
    df = pd.read_csv("./templates/medals.csv")
    All_Participants=Get_Count(df['athlete_full_name'])
    l1=list(All_Participants.keys())
    l2=list(All_Participants.values())
    # print("Number Of Times a Participant has Participated : ")
    # return l1
    return render(request, 'participant.html', {'l1':l1})


### 3 Finding Number of Participants in All Sports
def Get_Count(x):
  d=dict()
  for i in x:
    d[i]=d.get(i,0)+1
  return d
def get_sports_types(request):
    # df = pd.read_csv("./templates/medals.csv")
    All_Sports=Get_Count(df['discipline_title'])
    Sports_Types=list(All_Sports.keys())
    Sports_Types_Count=list(All_Sports.values())
    Sports_Types = {
        'Sports_Types': Sports_Types
    }
    return render(request,'game.html',Sports_Types)


# 5 Finding Number of Countries Participated
def country(request):
    All_Countries=Get_Count(df['country_name'])
    # print("All Countries Participated Till Now and its frequency is : ")
    c=list(All_Countries.keys())
    f=list(All_Countries.values())
    # content={'c':c , 'f': f}

    return render(request, 'country.html', {'c': c ,'f':f})

import pickle
from django.shortcuts import render
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the LSTM model (Assuming the model file is named 'lstm_model.h5')
model = load_model('lstm_model.h5')

# Function to preprocess the user input and make predictions
def medal(request):
    df = pd.read_csv("./templates/medals.csv")
    discipline_title = list(set(df['discipline_title']))
    event_gender = list(set(df['event_gender']))
    participant_type = list(set(df['participant_type']))
    if request.method == 'POST':
        discipline_title = request.POST.get('discipline_title', '')
        event_gender = request.POST.get('event_gender', '')
        participant_type = request.POST.get('participant_type', '')

        # Load the tokenizer (Assuming the tokenizer file is named 'tokenizer.pkl')
        with open('tokenizer.pkl', 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        # Preprocess the user input similar to the model training data
        text_data = discipline_title + ' ' + event_gender + ' ' + participant_type

        # Tokenize and pad the user input
        sequences = tokenizer.texts_to_sequences([text_data])
        max_length = 6  # Specify the maximum sequence length here (must be the same as during model training)
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

        # Make predictions using the loaded LSTM model
        y_pred_probs = model.predict(padded_sequences)
        predicted_class = 1 if y_pred_probs[0][0] >= 0.5 else 0

        # Map the predicted class to medal type
        medal_type = 'GOLD' if predicted_class == 1 else 'SILVER'

        return render(request, 'medal.html', {'discipline_title':discipline_title,'event_gender':event_gender,'participant_type':participant_type ,'medal_type': medal_type})
    else:
        return render(request, 'medal.html',{'discipline_title':discipline_title,'event_gender':event_gender,'participant_type':participant_type })
