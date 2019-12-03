#!/usr/bin/env python
# coding: utf-8

# # PEI  - Classification Model for NOS Technical problems

# ## Dataset
# We will use NOS private dataset with 197213 instances of techinal problems resolutions.

# ## Data Visualization , Preparation and Preprocessing

# In[1]:


import sys
import pandas as pd
import numpy as np

data = pd.read_csv('PEI_NOS_DATA.csv', sep=';', na_values=['NaN'],encoding='utf8')
data = data.sample(frac=1).reset_index(drop=True) # shuffle the data
print('Data size: ', data.shape)
data.head()


# In[2]:


data.describe()


# ### We will remove some clearly not important features for this problem! Selecting only closed cases!

# In[3]:


data = data[data['Estado_Registo_Siebel'] == "FECHADO"]
data = data[data.columns.difference(['Data_Hora_Criacao', 'Pagina_Saida', 'Descricao_Contexto_Saida', 'CA_key', 'UserKey_Current','Area_Key','Acao_Siebel','Estado_Registo_Siebel'])]
data.describe()


# ### Missing data
# Missing data is negligible, we can drop it.

# In[4]:


# to see the missing data context for some column
# data.loc[(data['Area_Key'].isnull())]
data.dropna(inplace=True)
data.describe()


# ### Target Cleaning

# In[5]:


data['Contexto_Saida'].value_counts().nlargest(n=15)


# In[6]:


print(data['Contexto_Saida'].describe(),"\n")
data = data[~data['Contexto_Saida'].str.lower().str.contains("despiste") == True ]
print(data['Contexto_Saida'].describe(),"\n")
data = data[~data['Tipificacao_Nivel_1'].str.lower().str.contains("despiste") == True ]
print(data['Contexto_Saida'].describe(),"\n")
data = data[~data['Tipificacao_Nivel_2'].str.lower().str.contains("despiste") == True ]
print(data['Contexto_Saida'].describe(),"\n")
data = data[~data['Tipificacao_Nivel_3'].str.lower().str.contains("despiste") == True ]
print(data['Contexto_Saida'].describe(),"\n")
data = data[~data['Contexto_Saida'].str.lower().str.strip().str.replace('ă', 'ã').str.contains("não resolv") == True ]
print(data['Contexto_Saida'].describe(),"\n")
data['Contexto_Saida'].value_counts().nlargest(n=15)


# In[7]:


import re
def clean_contexto_saida(st) :
    #str = "[Quebras wireless] - Hub;bloco Hard-eset resolve;;   ;;;;  ;;;22141241;Atinge velocade;reslvido;clt năo aceta propostas"
    st = st.lower().strip().replace('ă', 'ã').replace('ő','õ')
    comun_res =  re.search(r"((bloco configs)|(hard-reset))", st)
    if comun_res is not None:
        st = comun_res.group()
      #  st = re.sub(r"\W*","", st)

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


# In[8]:


s = "[Quebras wireless] - Hub;bloco Hard-eset resolve;;   ;;;;  ;;;22141241;Atinge velocade;reslvido;não pretende soluçőes"
clean_contexto_saida(s)


# In[9]:


for i in data.index:
    before = data.at[i, 'Contexto_Saida']
    after = clean_contexto_saida(before)
    data.at[i, 'Contexto_Saida'] = after
print(data['Contexto_Saida'].describe(),"\n")
data['Contexto_Saida'].value_counts().nlargest(n=15)


# ### Removing less frequent target's

# In[10]:


threshold = 150 # Anything that occurs less than this will be removed.
value_counts = data['Contexto_Saida'].value_counts() # Entire DataFrame 
to_remove = value_counts[value_counts <= threshold].index
data['Contexto_Saida'].replace(to_remove,'linhas apoio', inplace=True)
data.dropna(inplace=True)
data.groupby(['Contexto_Saida']).agg(['count'])


# In[11]:


data['Contexto_Saida'].describe()


# In[12]:


features = data[data.columns.difference(['Contexto_Saida'])]
target = data['Contexto_Saida']


# In[13]:


target.value_counts().nlargest(n=10)


# ### Data Discretization

# In[14]:


from sklearn import preprocessing
from collections import defaultdict

#features_test = features[features.columns.difference(['Tipificacao_Nivel_2','Tipificacao_Nivel_1'])]

d = defaultdict(preprocessing.LabelEncoder)
# Encoding the variable
features_encoded = features.apply(lambda x: d[x.name].fit_transform(x))

print(features_encoded.head())


# ### Target Balancing

# In[15]:


"""
unique, counts = np.unique(target, return_counts=True)
MAX_SIZE = 1000
ratio={}
i=0
while i<len(unique) :
    real_max = counts[i]
    if counts[i]>=MAX_SIZE :
        real_max=MAX_SIZE 
    ratio.update( {unique[i] : real_max } )
    i+=1

from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids

nm = NearMiss(ratio=ratio,version=1)
X_res, y_res = nm.fit_resample(features_encoded, target)

features_encoded = pd.DataFrame(X_res , columns =features.columns.tolist()) 
target = pd.Series(data = y_res )
"""


# In[16]:


target.describe()


# In[17]:


target.value_counts().nlargest(n=60)


# In[18]:


target.value_counts().nsmallest(n=10)


# In[19]:


import matplotlib.pyplot as plt
unique, counts = np.unique(target, return_counts=True)
plt.bar(unique, counts, 1)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


# ### Distribution of features

# In[20]:


features.groupby(['Origem_Despiste']).agg(['count'])


# In[21]:


features.groupby(['Servico']).agg(['count'])


# In[22]:


features.groupby(['Sintoma']).agg(['count'])


# In[23]:


features.groupby(['Tecnologia']).agg(['count'])


# In[24]:


features.groupby(['Tarifario']).agg(['count'])


# In[25]:


features.groupby(['Equipamento_Tipo']).agg(['count'])


# In[26]:


features.groupby(['Tipificacao_Nivel_1']).agg(['count'])


# In[27]:


features.groupby(['Tipificacao_Nivel_2']).agg(['count'])


# In[28]:


features.groupby(['Tipificacao_Nivel_3']).agg(['count'])


# ### Feature selection

# In[29]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold 

all_features_before = features_encoded
all_features = features_encoded


def features_scores(scores):
    i = 0
    for score in scores:
        print(features.columns.tolist()[i], " : ", scores[i]*100)
        i+=1

#Feature Selection - Univariate Selection
k=7
test = SelectKBest(score_func = mutual_info_classif, k = k)
fit = test.fit(all_features, target)
cols = test.get_support(indices=True)

# Sumarize Scores
features_scores(fit.scores_)
np.set_printoptions(precision=3)
all_features = fit.transform(all_features)
print(all_features.shape)


# ### Feature correlation

# In[30]:


import seaborn as sns

columns_all = features.columns.tolist()
columns = []
for i in cols :
    columns.append(columns_all[i])
    i+=1
    
fts = pd.DataFrame(data=all_features, 
                   columns=columns) 
def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v
fts.corr(method=histogram_intersection)
# calculate the correlation matrix
corr = fts.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# ## Training and Validation

# ### Number of features testing with Decision Tree

# In[31]:


"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=1)

cv_scores = cross_val_score(clf, all_features, target, cv=10)


# testing diferent numbers of features
i=5
for ft in features_list:
    cv_scores = cross_val_score(clf, ft, target, cv=10)
    print("\n\nNum of features = "+str(i))
    print('Accuracy', cv_scores.mean(),"\n\n")
    i+=1
"""


# ### Split train and test data

# In[32]:


import numpy
from sklearn.model_selection import train_test_split

d_target = defaultdict(preprocessing.LabelEncoder)
# Encoding the variable
target_encoded =  d_target['target'].fit_transform(target)

(training_inputs,testing_inputs,training_classes, testing_classes) = train_test_split(all_features, target_encoded , train_size=0.75, random_state=1)
num_classes = len(target.unique())

print('Num classes: ',num_classes)
print('training_input.shape: ', training_inputs.shape)
print('training_output.shape: ', training_classes.shape)
print('testing_input.shape: ', testing_inputs.shape)
print('testing_output.shape: ', testing_classes.shape)


# ### Target Distribution

# In[33]:


unique, counts = np.unique(training_classes, return_counts=True)
plt.bar(unique, counts)
unique, counts = np.unique(testing_classes, return_counts=True)
plt.bar(unique, counts)

plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.show()


# In[34]:


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


# ### Decision Tree

# In[35]:


from sklearn.tree import DecisionTreeClassifier

dt= DecisionTreeClassifier(random_state=1)
dt.fit(training_inputs, training_classes)


# In[36]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn import tree
from pydotplus import graph_from_dot_data 

dot_data = StringIO()  
tree.export_graphviz(dt, out_file=dot_data)  
                       #  feature_names=feature_names)  
graph = graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[37]:


all_scores=[]

score = dt.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### Random Forest

# In[38]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, random_state=1)
# Train the classifier on the training set
rf.fit(training_inputs, training_classes)
score = rf.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### SVM

# In[39]:


"""
from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear', C=C)
svc.fit(training_inputs, training_classes)
score = svc.score(testing_inputs, testing_classes)
all_scores.append(score)
score
"""


# ### KNN

# In[40]:


from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=10)
knn.fit(training_inputs, training_classes)
knn.score(testing_inputs, testing_classes)
best_score=0
for n in range(1, 50):
    knn = neighbors.KNeighborsClassifier(n_neighbors=n)
    knn.fit(training_inputs, training_classes)
    score = knn.score(testing_inputs, testing_classes)
    if (score>best_score) :
        best_score = score
    print (n, score)

all_scores.append(best_score)


# ### Naive Bayes

# In[41]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(training_inputs, training_classes)
score = nb.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### Logistic Regression

# In[42]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(training_inputs, training_classes)
score = lr.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### SGD

# In[43]:


from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(training_inputs, training_classes)
score = sgd.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### Neural network

# In[44]:


from keras.utils import to_categorical


training_classes_cat = to_categorical(training_classes)
testing_classes_cat = to_categorical(testing_classes)
training_classes_cat.shape


# In[45]:


from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def create_model(n_hidden,size_nodo,ativ,opt,dropout):
        model = Sequential()
        model.add(Dense(size_nodo, input_dim=k, activation=ativ))
        n=0
        for n in range(n_hidden) :
             model.add(Dropout(dropout))
             model.add(Dense(size_nodo, activation=ativ))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
# create estimator 
estimator = KerasClassifier(build_fn=create_model,
                            n_hidden=5,
                            size_nodo=12,
                            ativ="relu",
                            opt="adam",
                            dropout=0.1,
                            epochs=200,
                            batch_size=500,
                            verbose=1)
estimator.fit(training_inputs, training_classes_cat, validation_split=0.2, verbose=1)
score = estimator.score(testing_inputs,testing_classes_cat)
all_scores.append(score)
score


# ###  Overview  of all models

# In[46]:


print(all_scores)
print("Average of all models:",sum(all_scores)/len(all_scores))

# Last best = 0.73
# Last avg = 0.56

# tip1
# best = 0.51
# avg = 0.40

# tip2
# best = 0.57
# avg = 0.41

# tip3
# best = 0.69
# avg = 0.49


# ## Hyperparameters Optimization

# In[47]:


# todo


# # Save model function

# In[48]:


import pickle
def save_model(model,filename):
 pickle.dump(model, open(filename, 'wb'))


# # Open model

# In[49]:


def load_model(filename):
    model =pickle.load(open(filename, 'rb'))
    return model


# # Prediction

# In[50]:


print("Input Format:%s" % (columns))


# In[56]:


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


# ## Features options

# In[52]:


import json
import yaml

features_reverse_encoding = inverse_encoding(all_features.tolist())
def features_uniques(features) :
    features_uniques={}
    for key, value in features.iteritems(): 
        features_uniques.update({key : np.unique(features[key]).tolist() })
    return features_uniques

def save_dict_json(data,filename) : 
    with open(filename, 'w',encoding='utf8') as fp:
        json.dump(data, fp, indent=4, sort_keys=True,ensure_ascii=False)

input_options = features_uniques(features_reverse_encoding)
save_dict_json(input_options,"input_options.json")
input_options

            


# ## Trying to find relations between features

# In[53]:


features_reverse_encoding = inverse_encoding(all_features.tolist())

# versao 1
grouped_servico = features_reverse_encoding.groupby(['Servico'])
features_connections = {}
for servico, group_servico in grouped_servico:
  #  servico_unique = features_uniques(group_servico)
    grouped_sintoma = group_servico.groupby(['Sintoma'])
    
    sintoma_tips = {}
    for sintoma , group_sintoma in grouped_sintoma :
        tipificacoes = group_sintoma[['Tipificacao_Nivel_1','Tipificacao_Nivel_2','Tipificacao_Nivel_3']]
        grouped_tip1 = tipificacoes.groupby(['Tipificacao_Nivel_1'])
        tip1 = {}
        for tipificacao1 , group_tip1 in grouped_tip1 :
            group_tip1 = group_tip1[group_tip1.columns.difference(['Tipificacao_Nivel_1'])]
            grouped_tip2 = group_tip1.groupby(['Tipificacao_Nivel_2'])
            tip2 = {}
            for tipificacao2 , group_tip2 in grouped_tip2 :
                group_tip2 = group_tip2[group_tip2.columns.difference(['Tipificacao_Nivel_2'])]
                tip2.update({ tipificacao2 : features_uniques(group_tip2) })
            tip1.update({ tipificacao1 :  { 'Tipificacao_Nivel_2' : tip2 }  })
        tip = {'Tipificacao_Nivel_1' : tip1}
        sintoma_tips[sintoma] =  tip
    group_servico = group_servico[group_servico.columns.difference(['Servico','Sintoma','Tipificacao_Nivel_1','Tipificacao_Nivel_2','Tipificacao_Nivel_3'])]
    group_servico = features_uniques(group_servico)
    group_servico['Sintoma'] = sintoma_tips
    features_connections[servico] = group_servico 
input_options_related = {'Servico' : features_connections }
save_dict_json(input_options_related,"input_options_related_v1.json")

# versao 2
grouped_servico = features_reverse_encoding.groupby(['Servico'])
features_connections = {}
for servico, group_servico in grouped_servico:
  #  servico_unique = features_uniques(group_servico)
    grouped_sintoma = group_servico.groupby(['Sintoma'])
    group_servico = group_servico[group_servico.columns.difference(['Servico'])]
    group_servico = features_uniques(group_servico)
    features_connections[servico] = group_servico 
input_options_related = {'Servico' : features_connections }
save_dict_json(input_options_related,"input_options_related_v2.json")

# versao 3

tipificacoes = features_reverse_encoding[['Tipificacao_Nivel_1','Tipificacao_Nivel_2','Tipificacao_Nivel_3']]
grouped_tip1 = tipificacoes.groupby(['Tipificacao_Nivel_1'])
tip1 = {}
for tipificacao1 , group_tip1 in grouped_tip1 :
    group_tip1 = group_tip1[group_tip1.columns.difference(['Tipificacao_Nivel_1'])]
    grouped_tip2 = group_tip1.groupby(['Tipificacao_Nivel_2'])
    tip2 = {}
    for tipificacao2 , group_tip2 in grouped_tip2 :
        group_tip2 = group_tip2[group_tip2.columns.difference(['Tipificacao_Nivel_2'])]
        tip2.update({ tipificacao2 : features_uniques(group_tip2) })
    tip1.update({ tipificacao1 :  { 'Tipificacao_Nivel_2' : tip2 }  })

versao3 = features_reverse_encoding[features_reverse_encoding.columns.difference(['Tipificacao_Nivel_1','Tipificacao_Nivel_2','Tipificacao_Nivel_3'])]
versao3 = features_uniques(versao3)
versao3['Tipificacao_Nivel_1'] = tip1
save_dict_json(versao3,"input_options_related_v3.json")

# versao 4
tipificacoes = features_reverse_encoding[['Tipificacao_Nivel_1','Tipificacao_Nivel_2','Tipificacao_Nivel_3']]
grouped_tip3 = tipificacoes.groupby(['Tipificacao_Nivel_3'])
tip3 = {}
for tipificacao3 , group_tip3 in grouped_tip3 :
    group_tip3 = group_tip3[group_tip3.columns.difference(['Tipificacao_Nivel_3'])]
    grouped_tip2 = group_tip3.groupby(['Tipificacao_Nivel_2'])
    tip2 = {}
    for tipificacao2 , group_tip2 in grouped_tip2 :
        group_tip2 = group_tip2[group_tip2.columns.difference(['Tipificacao_Nivel_2'])]
        tip2.update({ tipificacao2 : features_uniques(group_tip2) })
    tip3.update({ tipificacao3 :  { 'Tipificacao_Nivel_2' : tip2 }  })

versao4 = features_reverse_encoding[features_reverse_encoding.columns.difference(['Tipificacao_Nivel_1','Tipificacao_Nivel_2','Tipificacao_Nivel_3'])]
versao4 = features_uniques(versao4)
versao4['Tipificacao_Nivel_3'] = tip3
save_dict_json(versao4,"input_options_related_v4.json")


# In[54]:


def predict_resolution(inputList,model):
    newInput = [ inputList ]
    input_encoded = encoding(newInput)
    input_encoded = input_encoded.values.tolist()

    model.fit(training_inputs, training_classes)
    score = model.score(testing_inputs, testing_classes)
    
    ynew = model.predict(input_encoded)
    probability = np.amax(model.predict_proba(input_encoded)) * 100
    return target_decoded(ynew)[0],probability


# In[55]:


# prediction input
save_model(rf,"model")
model = load_model("model")
newInput = [
              "HUB 3.0",
      #        "UDP",
              "Internet",
              "Velocidades Lentas",
              "Net Wideband Top - 120Mb",
       #       "HFC",
              "SERVIÇO INTERNET CABO",
              "LENTIDAO ACESSO WIRELESS",
              "DIFICULDADE DE UTILIZAÇĂO"
]
prediction,probability = predict_resolution(newInput,model)
print("X = %s\n\nPredicted = %s\n\nProbability = %.0f%%" % (newInput, prediction ,probability))

