#!/usr/bin/env python3
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import json, re, os
from threading import Thread
from keywords import keywords

embeddings = {}
embed = None

def loadModelData():
    ''' Loads Tensorflow enconder and pre-encodes the problem data
    '''
    global embed
    global embeddings

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2")
    feature_types = ['Sintoma', 'Tipificacao_Nivel_1', 'Tipificacao_Nivel_2', 'Tipificacao_Nivel_3']

    with open(os.getcwd() + '/input_options.json') as json_file:
        data = json.load(json_file)
        for typ in feature_types:
            embedProblemData(data, typ, embeddings)

            
def embedProblemData(data, feature_type, embeddings):
    ''' Calculates embeddings for all the values of feature_type
    :param: data
    :param: feature type
    :param: dict that maps feature values to their embeddings
    '''
    raw_features = [x for x in data[feature_type]]
    proc_features = [x.lower() for x in raw_features]
    feature_embeddings = embed(proc_features)["outputs"]
    for i in range(0, len(raw_features)):
        embeddings[raw_features[i]] = feature_embeddings[i]


def replaceWithKeywords(line, keywords):
    ''' Replaces matches in line with a keyword
    :param: string to look for expressions
    :param: dictionary object that matches keywords with expressions
    :return: list of versions of the line with replaced expressions
    '''
    keyworded_versions = [line]
    for keyword, matches in keywords.items():
        keyworded_versions.extend([re.sub(match, keyword, line) for match in matches if re.search(match, line)])

    return keyworded_versions


def getFeatureSuggestion(line, keywords, ss_vals, ss_embeddings, category):
    ''' Calculates feature from category that is semantically closest to the one described in
    line
    :param: target
    :param: 
    '''
    ll = line.lower()
    
    line_versions = replaceWithKeywords(ll, keywords['common'])
    if category>0:
        line_versions.extend(replaceWithKeywords(ll, keywords['tip_'+str(category)]))
        
    sentence_embeddings =  [embed(line_version)["outputs"] for line_version in line_versions]
    similarity_matrices = [list(np.inner(sent_emb, ss_embeddings)[0])
                           for sent_emb in sentence_embeddings]
    max_values = [max(similarity_matrice) for similarity_matrice in similarity_matrices]
    max_abs = max(max_values)
    similarity_matrix = similarity_matrices[max_values.index(max_abs)]
    sugestao = ss_vals[similarity_matrix.index(max_abs)]
    return sugestao, max_abs


def extractProblemData(prob_desc, search_space, category):
    ''' Extracts the string in the search space that is semantically 
    closest to the problem description
    :param: problem description
    :param: search space of the possible strings
    :param: search space category (simptome or typification)
    :return: closest string that belongs to search_space and confidence
    '''
    ss_embeddings = [embeddings[ss_val] for ss_val in search_space]
    return getFeatureSuggestion(prob_desc, keywords, search_space, ss_embeddings, category)
    
