import sys
import os
import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from collections import defaultdict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import pickle
import joblib
from timeit import default_timer as timer

def store_new_data(path_stored_new_data,path_latest_data) :
    """ Store new data into csv from bot users
    :param: Path to csv file of stored data from bot users
    :param: Path to csv file of new data from latest interations with bot
    :return: Pandas dataframe of the concatenated data
    """  
    latest_data = pd.read_csv(path_latest_data, sep=';', na_values=['NaN'],encoding='utf8',index_col=False)
    print("Latest data shape: " + str(latest_data.shape))
    if os.path.isfile(path_stored_new_data) :
        stored_data = pd.read_csv(path_stored_new_data, sep=';', na_values=['NaN'],encoding='utf8',index_col=False)
        print("Stored data shape: " + str(stored_data.shape))
        final_data = pd.concat([stored_data,latest_data],ignore_index=True,sort=False).reset_index(drop=True)
    else :
        print("No stored data found!!!")
        final_data = latest_data
    

    export_csv = final_data.to_csv(path_stored_new_data, index = None, header=True,sep=';')
    final_data = final_data.reset_index(drop=True)


    return final_data

def old_new_data_fusion(original_data,new_data) :
    print("Original Nos data shape: " + str(original_data.shape))
    print("Stored data + new data from bot   shape : " + str(new_data.shape))
    final_data = pd.concat([original_data,new_data],ignore_index=True,sort=False).reset_index(drop=True)
    print("Merge data shape: " + str(final_data.shape))
    final_data.dropna(inplace=True)
    print("Merge data shape after dropna: " + str(final_data.shape))
    return final_data

def load_data(filename):
    """ Get dataframe from csv file
    :param: Path to csv file
    :return: Pandas dataframe of the csv data
    """   
    data = pd.read_csv(filename, sep=';', na_values=['NaN'],encoding='utf8')
    data = data.sample(frac=1).reset_index(drop=True) # shuffle the data
    return data

def clean_contexto_saida(st) :
    """ Auxiliary method to extract problem resolution from Contexto_Saida column
    :param: String of interest (element of column Contexto_Saida)
    :return: String of problem resolution
    """   
    st = st.lower().strip().replace('ă', 'ã').replace('ő','õ')
    comun_res =  re.search(r"((bloco configs)|(hard-reset))", st)
    if comun_res is not None:
        st = comun_res.group()

    st = re.sub(r"\[.*\]\W*","", st)

    split_semicolon = re.split(";", st)
    split_hyphen = re.split("-", st)

    if len(split_semicolon) > len(split_hyphen)  :
        split = split_semicolon
    else : 
        split = split_hyphen

    n_split = len(split) 
    
    regex = re.compile(r"^((\s*fechado\s*)|(.*mantém dificuldade.*)|(.*hit ok.*)|(.*cliente informado.*)|(\W*outro equipamento.*)|(.*não pretende soluções.*)|(.*não aceita propostas.*)|(\W*resolv\w[\w\s]*)|(.*atinge velocidade.*)|(^\d*$)|(^\W*$)|(.*aceita proposta sugerida\W*)|(\W*loja\W*)|(.*não consegue realizar procedimento\W*))$")
    filtered = [i for i in split if not regex.match(i)]
    if (len(filtered)>0) :
        result = filtered[ len(filtered) - 1 ]
        if ('fora de portugal continental' in result) :
            result = 'pesquisa canais'
        if ('box bloqueada' in result) :
            result = 'dvl box ou repor parâmetros parcial painel'
        if ('hitron' in result) or ('sintoma direitos' in result) :
            return np.nan
        return str(result).strip()
    else :
        return np.nan

def clean_originaldata(data):
    """ Method that makes several changes to the original data from Nos dataframe to make it ready for training, removes unnecessary columns, etc
    :param: Dataframe from original Nos csv data
    :return: Dataframe cleaned
    """  
    data = data[data['Estado_Registo_Siebel'] == "FECHADO"]
    data = data[data.columns.difference(['Data_Hora_Criacao', 'Pagina_Saida', 'Descricao_Contexto_Saida', 'CA_key', 'UserKey_Current','Area_Key','Acao_Siebel','Estado_Registo_Siebel'])]
    data.dropna(inplace=True)
    data = data[~data['Contexto_Saida'].str.lower().str.contains("despiste") == True ]
    data = data[~data['Tipificacao_Nivel_1'].str.lower().str.contains("despiste") == True ]
    data = data[~data['Tipificacao_Nivel_2'].str.lower().str.contains("despiste") == True ]
    data = data[~data['Tipificacao_Nivel_3'].str.lower().str.contains("despiste") == True ]
    data = data[~data['Contexto_Saida'].str.lower().str.strip().str.replace('ă', 'ã').str.contains("não resolv") == True ]
    for i in data.index:
        before = data.at[i, 'Contexto_Saida']
        after = clean_contexto_saida(before)
        data.at[i, 'Contexto_Saida'] = after
    threshold = 150 # Anything that occurs less than this will be support lines.
    value_counts = data['Contexto_Saida'].value_counts()
    to_remove = value_counts[value_counts <= threshold].index
    data['Contexto_Saida'].replace(to_remove,'linhas apoio', inplace=True)
    data = data[['Sintoma','Tipificacao_Nivel_1','Tipificacao_Nivel_2','Tipificacao_Nivel_3','Equipamento_Tipo','Servico','Tarifario','Contexto_Saida' ]]
    return data


def target_selection(data) :
    """ Separates features from target
    :param: Dataframe
    :return: Features pandas.Dataframe
    :return: Target pandas.Series
    """ 
    data = data[['Sintoma','Tipificacao_Nivel_1','Tipificacao_Nivel_2','Tipificacao_Nivel_3','Equipamento_Tipo','Servico','Tarifario','Contexto_Saida' ]]
    features = data[data.columns.difference(['Contexto_Saida'])]
    target = data['Contexto_Saida']
    return features,target

def data_discretization(data_to_encode,multilplecolumns=True):
    """ Discretization of dataframe or series using preprocessing.LabelEncoder
    :param: Fataframe or series to encode
    :param: Bolean to determinate if data is dataframe or series
    :return: Encoded dataframe or series
    :return: Dictionary needed to reverse encoding, or encode new data
    """
    d = defaultdict(preprocessing.LabelEncoder)
    if multilplecolumns :
        encoded = data_to_encode.apply(lambda x: d[x.name].fit_transform(x))
    else :
        encoded =  d['target'].fit_transform(data_to_encode)
    return encoded,d

def training_setup(data):
    """ Setup training and testing data, including discretization , also saves dictionarys of discretization
    :param: Fataframe to setup
    :return: Features data for training
    :return: Features data for testing
    :return: Target data for training
    :return: Target data for testing
    :return: Features encoded
    :return: Target encoded
    :return: Number of different classes in target
    """
    features,target = target_selection(data)
    features_encoded,d = data_discretization(features,True)
    target_encoded,d_target = data_discretization(target,False)
    save_dict(d,path + 'features_dict')
    save_dict(d_target,path + 'target_dict')

    (training_inputs,testing_inputs,training_classes, testing_classes) = train_test_split(features_encoded, target_encoded , train_size=0.75, random_state=1)
    num_classes = len(target.unique())
    print('Num classes: ',num_classes)
    print('training_input.shape: ', training_inputs.shape)
    print('training_output.shape: ', training_classes.shape)
    print('testing_input.shape: ', testing_inputs.shape)
    print('testing_output.shape: ', testing_classes.shape)
    return training_inputs,testing_inputs,training_classes, testing_classes , features_encoded , target_encoded , num_classes

def target_distribution_plot(training_classes,testing_classes):
    """ Plot showing distribution of classes of training and testing inputs
    :param: Fataframe
    :return: Features pandas.Dataframe
    :return: Target pandas.Series
    """ 
    unique, counts = np.unique(training_classes, return_counts=True)
    plt.bar(unique, counts)
    unique, counts = np.unique(testing_classes, return_counts=True)
    plt.bar(unique, counts)

    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Frequency')

    plt.show()

    total_training = len(training_classes)
    total_testing = len(testing_classes)
    unique, counts = np.unique(training_classes, return_counts=True)
    plt.bar(unique, counts/total_training)
    plt.title('Class Frequency Training %')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

    unique, counts = np.unique(testing_classes, return_counts=True)
    plt.bar(unique, counts/total_testing)
    plt.title('Class Frequency Testing %')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

def grid_search(X, y, nfolds,param_grid,model):
    """ Auxiliary method to implement grid search algoritm for hyperparameter tuning 
    :param: Features
    :param: Target
    :param: Nfolds for cross validation
    :param: Grid of hyperparameters to test
    :param: Model to find best hyperparameters
    :return: Best hyperparameters
    """ 
    global n_jobs_global
    grid_search = GridSearchCV(model, param_grid, cv=nfolds, verbose=2, n_jobs=n_jobs_global)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def models_training_and_evaluating(training_inputs,testing_inputs,training_classes, testing_classes,models):
    """ Train and evaluate all models
    :param: Features data for training
    :param: Features data for testing
    :param: Target data for training
    :param: Target data for testing
    :param: Models dictionary
    :return: Models dictionary with scores updated for each model
    """
    print("Number of models: " + str(len(models)))
    for model in models :
        model.fit(training_inputs, training_classes)
        score = model.score(testing_inputs, testing_classes)
        models[model] = score
        print("=>>>>>>")
        print(model)
        print("Score : " + str(score))
    return models

def KNeighbors (training_inputs,testing_inputs,training_classes, testing_classes) :
    """ Find best number of neighbors on KNeighbors classifier
    :param: Teatures data for training
    :param: Features data for testing
    :param: Target data for training
    :param: Target data for testing
    :return: KNeighbors model with best parameters 
    """
    best_model = neighbors.KNeighborsClassifier(n_neighbors=10)
    best_score=0
    for n in range(1, 50):
        knn = neighbors.KNeighborsClassifier(n_neighbors=n)
        knn.fit(training_inputs, training_classes)
        score = knn.score(testing_inputs, testing_classes)
        if (score>best_score) :
            best_score = score
            best_model = knn
    return best_model

def rf_hyper( features_encoded , target_encoded) :
    """ Hyperparameter tuning of Random Forest classifier
    :param: Features data
    :param: Target data
    :return: Random Forest model with best parameters 
    """
    global hyperparameter
    if not hyperparameter :
        return RandomForestClassifier(n_estimators=10, random_state=1)

    n_estimators = [100,250,400]
    max_depth = [8,12,14,None]
    random_state = [1]
    max_features = ['auto']
    param_grid = {'n_estimators': n_estimators, 'max_depth' : max_depth, 'random_state' : random_state, 'max_features' : max_features}

    rf_hyper_parameters = grid_search(features_encoded, target_encoded, 2 ,  param_grid, RandomForestClassifier())
    print('\n\n\nBest RF Hyper-parameters using GridSearch:\n', rf_hyper_parameters)

    n_estimators = rf_hyper_parameters['n_estimators']
    max_depth = rf_hyper_parameters['max_depth']
    random_state = rf_hyper_parameters['random_state']
    max_features = rf_hyper_parameters['max_features']

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth, max_features=max_features)
    return rf

def lr_hyper( features_encoded , target_encoded) :
    """ Hyperparameter tuning of Logistic Regression classifier
    :param: Teatures data
    :param: Target data
    :return: Logistic Regression model with best parameters 
    """
    global hyperparameter
    if not hyperparameter :
        return LogisticRegression()

    param_grid={
            "C":np.logspace(-3,3,5), 
            "penalty":["l1","l2"]
    }
    lr_hyper_parameters = grid_search(features_encoded, target_encoded, 2 ,param_grid ,LogisticRegression())
    print('\n\n\nBest Logistic Regression Hyper-parameters using GridSearch:\n', lr_hyper_parameters)

    C = lr_hyper_parameters['C']
    penalty = lr_hyper_parameters['penalty']

    return LogisticRegression(C=C,penalty=penalty)

def sgd_hyper( features_encoded , target_encoded) : 
    """ Hyperparameter tuning of SGD classifier
    :param: Features data
    :param: Target data
    :return: SGD model with best parameters 
    """
    global hyperparameter
    if not hyperparameter :
        return SGDClassifier()
    params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "none"],
    }
    sgd_hyper_parameters = grid_search(features_encoded, target_encoded, 2 , params , SGDClassifier())
    print('\n\n\nBest SGD Hyper-parameters using GridSearch:\n', sgd_hyper_parameters)

    loss = sgd_hyper_parameters['loss']
    alpha = sgd_hyper_parameters['alpha']
    penalty = sgd_hyper_parameters['penalty']

    sgd = SGDClassifier(loss=loss,alpha=alpha,penalty=penalty)

def create_model(n_hidden,size_nodo,ativ,opt,dropout):
    """ Neural network model setup method
    :param: Number of hidden layers
    :param: Number of nodos on each layer
    :param: Activation function
    :param: Optimizer
    :param: Dropout chance (0 - 1)
    :return: Neural network model
    """
    global num_classes
    model = Sequential()
    model.add(Dense(size_nodo, input_dim=7, activation=ativ))
    n=0
    for n in range(n_hidden) :
            model.add(Dropout(dropout))
            model.add(Dense(size_nodo, activation=ativ))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def neural_network_hyper( features_encoded , target_encoded) :
    """ Hyperparameter tuning of Neural network
    :param: Features data
    :param: Target data
    :return: Neural network model with best parameters 
    """
    global hyperparameter
    if not hyperparameter :
        estimator = KerasClassifier(build_fn=create_model,
                            n_hidden=5,
                            size_nodo=12,
                            ativ="relu",
                            opt="adam",
                            dropout=0.1,
                            epochs=500,
                            batch_size=10000,
                            validation_split=0.1,
                            verbose=0)
        return estimator
    grid_param = {  
        'n_hidden': [2,5],
        'size_nodo': [50,200],
        'ativ':['relu','softmax'],
        'opt': ['adam'],
        'dropout' : [0.1],
        'epochs' : [50],
        'batch_size' : [20000]
    }
    model = KerasClassifier(build_fn=create_model,verbose=1,validation_split=0.1)
    ann_hyper_parameters = grid_search(features_encoded, target_encoded, 2 ,  grid_param, model)
    print('\n\n\nBest Neural Network Hyper-parameters using GridSearch:\n', ann_hyper_parameters)


    estimator = KerasClassifier(build_fn=create_model,
                                n_hidden=ann_hyper_parameters['n_hidden'],
                                size_nodo=ann_hyper_parameters['size_nodo'],
                                ativ=ann_hyper_parameters['ativ'],
                                opt=ann_hyper_parameters['opt'],
                                dropout=ann_hyper_parameters['dropout'],
                                epochs=1000, #ann_hyper_parameters['epochs'],
                                batch_size=ann_hyper_parameters['batch_size'],
                                validation_split=0.1,
                                verbose=1)
    return estimator

def select_best_model(models):
    """ Selection best model based on score
    :param: Models dictionary
    :return: Best model
    :return: Accuracy score of model
    """
    best_model = "none"
    max_score = 0
    for model in models:
        if models[model] > max_score :
            max_score = models[model]
            best_model = model
    return best_model,max_score

def save_model(model,filename):
    """ Save classification model in file
    :param: Model to save
    :param: Path
    """
    pickle.dump(model, open(filename, 'wb'))

def save_dict(dict_encoder, filename) :
    """ Save dictionary of encoded data
    :param: Dictionary
    :param: Path
    """
    joblib.dump(dict_encoder,filename)

def dynamically_training(path_Nos_originaldata, path_stored_new_data, path_latest_data ,path_to_output, plots=False,hyperparameter_tuning = False,n_jobs=1) : 
    """ Executing everything needed to load, clean, test , train, evalutate models, and choose the best to save
    :param: Path to dataset in csv with the original data from nos
    :param: Path to dataset in csv with the stored data from our bot old interations, its created if doesnt exist
    :param: Path to dataset in csv with the latest data given by users of our bot
    :param: Path to where we want to store the final model and dictionarys
    :param: Boolean indicating if we want to see plots of distribution of classes
    :param: Boolean indicating if we want hyperparameter tuning of models, which can require several hours
    :param: The number of jobs to run in parallel, -1 means using all processors
    """
    global path,num_classes,hyperparameter,n_jobs_global
    n_jobs_global=n_jobs
    start = timer()
    hyperparameter = hyperparameter_tuning
    path = path_to_output

    new_data = store_new_data(path_stored_new_data,path_latest_data)
    
    original_data = load_data(path_Nos_originaldata)
    original_data = clean_originaldata(original_data)
    original_data.dropna(inplace=True)

    data = old_new_data_fusion(original_data,new_data)

    training_inputs,testing_inputs,training_classes, testing_classes, features_encoded , target_encoded, num_classes = training_setup(data)
    if plots :
         target_distribution_plot(training_classes,testing_classes)
    models = {
        DecisionTreeClassifier(random_state=1) : 0,
        svm.SVC(kernel='linear', C=1.0) : 0 ,
        KNeighbors (training_inputs,testing_inputs,training_classes, testing_classes) : 0 ,
        MultinomialNB() : 0 ,
        rf_hyper( features_encoded , target_encoded) : 0,
        lr_hyper( features_encoded , target_encoded) : 0 ,
        neural_network_hyper( features_encoded , target_encoded) : 0
    }

    models = models_training_and_evaluating(training_inputs,testing_inputs,training_classes, testing_classes,models)

    best_model,max_score = select_best_model(models)

    print("\n\nBest_Model : " + str(best_model))
    if (best_model.get_params()):
        print("\n\nParams : " + str(best_model.get_params()))
    print("Score : " + str(max_score))

    save_model(best_model,path + "model")
    print("Model Saved !!! ")
    end = timer()
    print("\n\nDuration of training in minutes : " +  str((end - start)/60) + " minutes." )


path = "./"
num_classes = 0
hyperparameter = False
n_jobs_global = 1

#dynamically_training(path_Nos_originaldata='PEI_NOS_DATA.csv', path_stored_new_data='stored.csv'  ,path_latest_data='log.csv',path_to_output="./",plots=False, hyperparameter_tuning = False,n_jobs=2)
