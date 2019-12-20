import pickle
import numpy as np
import joblib
import pandas as pd
import os

def load_dict(filename) :    
    label_encoder = joblib.load(filename)
    return label_encoder

d = load_dict(os.getcwd() + '/technical_problems/model_files/features_dict')
d_target = load_dict(os.getcwd() + '/technical_problems/model_files/target_dict')
columns = ['Equipamento_Tipo', 'Servico', 'Sintoma', 'Tarifario', 'Tipificacao_Nivel_1', 'Tipificacao_Nivel_2', 'Tipificacao_Nivel_3']

def load_model(filename):
    model =pickle.load(open(filename, 'rb'))
    return model

def encoding(inpArray):
    newInputDf = pd.DataFrame(inpArray , columns =columns) 
    input_encoded =newInputDf.apply(lambda x: d[x.name].transform(x))
    return input_encoded

def inverse_encoding(inpArray):
    newInputDf = pd.DataFrame(inpArray , columns =columns) 
    input_reversed =newInputDf.apply(lambda x: d[x.name].inverse_transform(x))
    return input_reversed

def target_decoded(target):
    target_decoded =  d_target['target'].inverse_transform(target)
    return target_decoded

def predict_resolution(inputList,model):
    newInput = [ inputList ]
    input_encoded = encoding(newInput)
    input_encoded = input_encoded.values.tolist()
    
    ynew = model.predict(input_encoded)
    probability = np.amax(model.predict_proba(input_encoded)) * 100
    return target_decoded(ynew)[0],probability