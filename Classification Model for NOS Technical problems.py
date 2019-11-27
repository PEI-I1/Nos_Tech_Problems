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


# In[52]:


import re

def clean_contexto_saida(st) :
    #str = "[Quebras wireless] - Hub;bloco Hard-eset resolve;;   ;;;;  ;;;22141241;Atinge velocade;reslvido;clt năo aceta propostas"
    st = st.lower().strip().replace('ă', 'ã')
    comun_res =  re.search(r"((dvl)|(bloco configs)|(hard-reset))", st)
    if comun_res is not None:
        st = comun_res.group()
        st = re.sub(r"\W*","", st)

    st = re.sub(r"\[.*\]\W*","", st)

    split_semicolon = re.split(";", st)
    split_hyphen = re.split("-", st)

    if len(split_semicolon) > len(split_hyphen)  :
        split = split_semicolon
    else : 
        split = split_hyphen

    n_split = len(split) 

    regex = re.compile(r"^((fechado)|(clt não aceita propostas)|(resolv\w+)|(atinge velocidade)|(^\d*$)|(^\W*$))$")
    filtered = [i for i in split if not regex.match(i)]
    result = ''
    if (len(filtered)>0) :
        result = filtered[ len(filtered) - 1 ]
    return result


# In[9]:


for i in data.index:
    before = data.at[i, 'Contexto_Saida']
    after = clean_contexto_saida(before)
    data.at[i, 'Contexto_Saida'] = after
print(data['Contexto_Saida'].describe(),"\n")
data['Contexto_Saida'].value_counts().nlargest(n=15)


# ### Removing less frequent target's

# In[10]:


threshold = 20 # Anything that occurs less than this will be removed.
value_counts = data.stack().value_counts() # Entire DataFrame 
to_remove = value_counts[value_counts <= threshold].index
data.replace(to_remove, np.nan, inplace=True)
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
d = defaultdict(preprocessing.LabelEncoder)

# Encoding the variable
features_encoded = features.apply(lambda x: d[x.name].fit_transform(x))

print(features_encoded.head())


# ### More Target Balancing -> Todo

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


target.value_counts().nlargest(n=10)


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
    

# to test later diferent numbers of features
i=5
features_list=[]
while i<=8:
    print("Num Features:",i)
    n=i
    test = SelectKBest(score_func = mutual_info_classif, k = n)
    fit = test.fit(all_features_before, target)
    # Sumarize Scores
    np.set_printoptions(precision=3)
    features_list.append(fit.transform(all_features_before))
    i+=1


# ## Training and Validation

# ### Number of features testing with Decision Tree

# In[30]:


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


# ### Split train and test data

# In[31]:


import numpy
from sklearn.model_selection import train_test_split
(training_inputs,testing_inputs,training_classes, testing_classes) = train_test_split(all_features, target, train_size=0.75, random_state=1)

print('training_input.shape: ', training_inputs.shape)
print('training_output.shape: ', training_classes.shape)
print('testing_input.shape: ', testing_inputs.shape)
print('testing_output.shape: ', testing_classes.shape)


# ### Target Distribution

# In[32]:


unique, counts = np.unique(training_classes, return_counts=True)
plt.bar(unique, counts)
unique, counts = np.unique(testing_classes, return_counts=True)
plt.bar(unique, counts)

plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.show()


# In[33]:


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

# In[34]:


from sklearn.tree import DecisionTreeClassifier

dt= DecisionTreeClassifier(random_state=1)
dt.fit(training_inputs, training_classes)


# In[35]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn import tree
from pydotplus import graph_from_dot_data 

dot_data = StringIO()  
tree.export_graphviz(dt, out_file=dot_data)  
                       #  feature_names=feature_names)  
graph = graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[36]:


all_scores=[]

score = dt.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### Random Forest

# In[37]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, random_state=1)
# Train the classifier on the training set
rf.fit(training_inputs, training_classes)
score = rf.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### SVM

# In[38]:



from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear', C=C)
svc.fit(training_inputs, training_classes)
score = svc.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### KNN

# In[39]:


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

# In[40]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(training_inputs, training_classes)
score = nb.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### Logistic Regression

# In[41]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(training_inputs, training_classes)
score = lr.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### SGD

# In[42]:


from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(training_inputs, training_classes)
score = sgd.score(testing_inputs, testing_classes)
all_scores.append(score)
score


# ### Neural network

# In[43]:


# todo


# ###  Overview  of all models

# In[44]:


print(all_scores)
print("Average of all models:",sum(all_scores)/len(all_scores))
# Last best = 0.67
# Last avg = 0.57


# ## Hyperparameters Optimization

# In[45]:


# todo


# # Save model function

# In[46]:


import pickle
def save_model(model,filename):
 pickle.dump(model, open(filename, 'wb'))


# # Open model

# In[47]:


def load_model(filename):
    model =pickle.load(open(filename, 'rb'))
    return model


# # Prediction

# In[48]:


columns_all = features.columns.tolist()
columns = []
for i in cols :
    columns.append(columns_all[i])
    i+=1
print("Input Format:%s" % (columns))


# In[49]:


import json

def features_uniques(features) :
    features_uniques={}
    for key, value in features.iteritems(): 
        features_uniques.update({key : np.unique(features[key]).tolist() })
    return features_uniques

def save_dict_json(data,filename) : 
    with open(filename, 'w',encoding='utf8') as fp:
        json.dump(data, fp, indent=4, sort_keys=True,ensure_ascii=False)

input_options = features_uniques(features)
save_dict_json(input_options,"input_options.json")
input_options


# In[50]:


def predict_resolution(inputList,model):
    newInput = [ inputList ]
    newInputDf = pd.DataFrame(newInput , columns =columns) 
    input_encoded =newInputDf.apply(lambda x: d[x.name].transform(x))
    input_encoded = input_encoded.values.tolist()

    model.fit(training_inputs, training_classes)
    score = model.score(testing_inputs, testing_classes)
    
    # make a prediction
    ynew = model.predict(input_encoded)
    probability = np.amax(model.predict_proba(input_encoded)) * 100
    return ynew[0],probability


# In[51]:


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

