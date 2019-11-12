#!/usr/bin/env python
# coding: utf-8

# # PEI  - Classification Model for NOS Technical problems

# ## Dataset
# We will use NOS private dataset with 197213 instances of techinal problems resolutions.

# ## Data Visualization , Preparation and Preprocessing

# In[ ]:


import sys
import pandas as pd
import numpy as np

data = pd.read_csv('PEI_NOS_DATA.csv', sep=';', na_values=['NaN'])
data = data.sample(frac=1).reset_index(drop=True) # shuffle the data
print('Data size: ', data.shape)
data.head()


# In[ ]:


data.describe()


# We will remove some clearly not important features for this problem!

# In[ ]:


data = data[data['Estado_Registo_Siebel'] == "FECHADO"]
data = data[data.columns.difference(['Data_Hora_Criacao', 'Contexto_Saida', 'Descricao_Contexto_Saida', 'CA_key', 'UserKey_Current','Area_Key','Acao_Siebel','Estado_Registo_Siebel'])]
data.describe()


# ### Missing data
# Missing data is negligible, we can drop it.

# In[ ]:


# to see the missing data context for some column
# data.loc[(data['Area_Key'].isnull())]
data.dropna(inplace=True)
data.describe()


# Distribution of target "Pagina_Saida" and other features

# In[ ]:


data.groupby(['Pagina_Saida']).agg(['count'])


# In[ ]:


data.groupby(['Origem_Despiste']).agg(['count'])


# In[ ]:


data.groupby(['Servico']).agg(['count'])


# In[ ]:


data.groupby(['Sintoma']).agg(['count'])


# In[ ]:


data.groupby(['Tecnologia']).agg(['count'])


# In[ ]:


data.groupby(['Tipificacao_Nivel_1']).agg(['count'])


# In[ ]:


data.groupby(['Tipificacao_Nivel_2']).agg(['count'])


# In[ ]:


data.groupby(['Tipificacao_Nivel_3']).agg(['count'])


# In[ ]:


features = data[data.columns.difference(['Pagina_Saida'])]
target = data['Pagina_Saida']


# ### Data Discretization

# In[ ]:


from sklearn import preprocessing
from collections import defaultdict
d = defaultdict(preprocessing.LabelEncoder)

# Encoding the variable
features_encoded = features.apply(lambda x: d[x.name].fit_transform(x))

#scaler = preprocessing.LabelEncoder()
#features_encoded = features.apply(scaler.fit_transform)
#features_encoded.head()
print(features_encoded.head())


# ### Feature selection

# In[ ]:


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
k=9
test = SelectKBest(score_func = mutual_info_classif, k = k)
fit = test.fit(all_features, target)
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

# In[ ]:


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


# ### Decision Tree

# In[ ]:


import numpy
from sklearn.model_selection import train_test_split
(training_inputs,testing_inputs,training_classes, testing_classes) = train_test_split(all_features, target, train_size=0.75, random_state=1)

print('training_input.shape: ', training_inputs.shape)
print('training_output.shape: ', training_classes.shape)
print('testing_input.shape: ', testing_inputs.shape)
print('testing_output.shape: ', testing_classes.shape)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt= DecisionTreeClassifier(random_state=1)
dt.fit(training_inputs, training_classes)


# In[ ]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn import tree
from pydotplus import graph_from_dot_data 

dot_data = StringIO()  
tree.export_graphviz(dt, out_file=dot_data)  
                       #  feature_names=feature_names)  
graph = graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[ ]:


dt.score(testing_inputs, testing_classes)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, random_state=1)
# Train the classifier on the training set
rf.fit(training_inputs, training_classes)
rf.score(testing_inputs, testing_classes)


# ### SVM

# In[ ]:


from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear', C=C)
svc.fit(training_inputs, training_classes)
svc.score(testing_inputs, testing_classes)


# ### KNN

# In[ ]:


from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=10)
knn.fit(training_inputs, training_classes)
knn.score(testing_inputs, testing_classes)
for n in range(1, 50):
    knn = neighbors.KNeighborsClassifier(n_neighbors=n)
    knn.fit(training_inputs, training_classes)
    score = knn.score(testing_inputs, testing_classes)
    print (n, score)


# ### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(training_inputs, training_classes)
nb.score(testing_inputs, testing_classes)


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(training_inputs, training_classes)
lr.score(testing_inputs, testing_classes)


# ### SGD

# In[ ]:


from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(training_inputs, training_classes)
sgd.score(testing_inputs, testing_classes)


# ### Neural network

# In[ ]:


# todo


# ## Hyperparameters Optimization

# In[ ]:


# todo


# ## Prediction

# In[ ]:


print("Input Format:%s" % (features.columns.tolist()))


# In[ ]:


# prediction input
model = clf
newInput = [[
              "HUB 3.0",
              "UDP",
              "Voz",
              "Sem Acesso",
              "Phone Ilimitado",
              "HFC",
              "SERVIÇO INTERNET CABO",
              "SEM ACESSO",
              "DESPISTE INTERROMPIDO"
]]
newInputDf = pd.DataFrame(newInput , columns =features.columns.tolist()) 
input_encoded =newInputDf.apply(lambda x: d[x.name].transform(x))
input_encoded = input_encoded.values.tolist()

model.fit(all_features, target)
# make a prediction
ynew = model.predict(input_encoded)
print("X=%s\n\nX_encoded=%s\n\nPredicted=%s" % (newInput[0],input_encoded[0], ynew[0]))


# In[ ]:




