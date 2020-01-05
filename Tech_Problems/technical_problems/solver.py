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

def best_n_suggestions(probs,n=3):
    dic = dict(enumerate(probs.flatten(), 0))
    sorted_dict = sorted(dic.items(), key=lambda item: item[1])
    best_suggests= []
    best_suggests_probs =[]
    i=1
    while i<=n:
        new_prob = sorted_dict[len(sorted_dict)-i][1]
        if new_prob > 0 :
            best_suggests.append(sorted_dict[len(sorted_dict)-i][0])
            best_suggests_probs.append(new_prob)
        i+=1
    return best_suggests,best_suggests_probs

def predict_resolution(inputList,model):
    newInput = [ inputList ]
    input_encoded = encoding(newInput)
    input_encoded = input_encoded.values.tolist()
    probs = model.predict_proba(input_encoded)[0]
    suggests,probs = best_n_suggestions(model.predict_proba(input_encoded) , 3)
    return list(zip(suggests,probs))
