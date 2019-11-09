#!/usr/bin/env python
# coding: utf-8

# # PEI  - Classification Model for NOS Technical problems

# ## Dataset
# We will use NOS private dataset with 197213 instances of techinal problems resolutions.

# ## Data Visualization , Preparation and Preprocessing

# In[39]:


import sys
import pandas as pd
import numpy as np

data = pd.read_csv('PEI_NOS_DATA.csv', sep=';', na_values=['NaN'])

print('Data size: ', data.shape)
data.head()


# In[40]:


data.describe()


# We will remove some clearly not important features for this problem!

# In[41]:


data = data[data.columns.difference(['Data_Hora_Criacao', 'Contexto_Saida', 'Descricao_Contexto_Saida', 'CA_key', 'UserKey_Current','Area_Key'])]
data = data[data['Estado_Registo_Siebel'] == "FECHADO"]
data.describe()


# ### Missing data
# Missing data is negligible, we can drop it.

# In[42]:


# to see the missing data context for some column
# data.loc[(data['Area_Key'].isnull())]
data.dropna(inplace=True)
data.describe()


# Distribution of target "Pagina_Saida" and other features

# In[43]:


data.groupby(['Pagina_Saida']).agg(['count'])


# In[44]:


data.groupby(['Acao_Siebel']).agg(['count'])


# In[45]:


data.groupby(['Estado_Registo_Siebel']).agg(['count'])


# In[46]:


data.groupby(['Origem_Despiste']).agg(['count'])


# In[47]:


data.groupby(['Servico']).agg(['count'])


# In[48]:


data.groupby(['Sintoma']).agg(['count'])


# In[49]:


data.groupby(['Tecnologia']).agg(['count'])


# In[50]:


data.groupby(['Tipificacao_Nivel_1']).agg(['count'])


# In[51]:


data.groupby(['Tipificacao_Nivel_2']).agg(['count'])


# In[52]:


data.groupby(['Tipificacao_Nivel_3']).agg(['count'])


# In[53]:


features = data[data.columns.difference(['Pagina_Saida'])]
target = data['Pagina_Saida']


# ### Data Normalization

# In[54]:


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

# In[55]:


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
k=11
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
while i<=10:
    print("Num Features:",i)
    n=i
    test = SelectKBest(score_func = mutual_info_classif, k = n)
    fit = test.fit(all_features_before, target)
    # Sumarize Scores
    np.set_printoptions(precision=3)
    features_list.append(fit.transform(all_features_before))
    i+=1


# ## Training and Validation

# In[56]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=1)

cv_scores = cross_val_score(clf, all_features, target, cv=10)


# to fit with  diferent numbers of features
i=5
for ft in features_list:
    cv_scores = cross_val_score(clf, ft, target, cv=10)
    print("\n\nNum of features = "+str(i))
    print('Accuracy', cv_scores.mean(),"\n\n")
    i+=1


# ## Hyperparameters Optimization

# In[ ]:





# ## Prediction

# In[58]:


print("Input Format:%s" % (features.columns.tolist()))


# In[59]:


# prediction input
model = clf
newInput = [[
              "Avaria Individual",
              "HUB 3.0",
              "FECHADO",
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




