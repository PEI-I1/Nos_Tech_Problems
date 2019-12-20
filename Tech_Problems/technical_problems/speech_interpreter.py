#!/usr/bin/env python3
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import json, re, os
from threading import Thread
from .keywords import keywords

emb_og_typifications = []
embed = None
result_sintoma = None
sintoma = None

def loadModelData():
    ''' Loads Tensorflow enconder and pre-encodes the typification sentences
    '''
    global embed
    global result_sintoma
    global sintoma
    global emb_og_typifications

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2")
    result_sintoma, sintoma = loadProblemData('Sintoma')
    for i in range(1,4):
        emb_og_typifications.append(loadProblemData('Tipificacao_Nivel_'+str(i)))
    print('-------------> READY <-------------')


def loadProblemData(feature_name):
    ''' Load typifications and encode them
    '''
    with open(os.getcwd() + '/technical_problems/model_files/input_options.json') as json_file:
        data = json.load(json_file)
        feature_og = [x for x in data[feature_name]]
        feature = [x.lower() for x in feature_og]
    return embed(feature)["outputs"], feature_og


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


def getFeatureSuggestion(line, keywords, result_tipificacao, tipificacao_tipo, response, index):
    ll = line.lower()
    line_versions = replaceWithKeywords(ll, keywords['common'])
    if index>0:
        line_versions.extend(replaceWithKeywords(ll, keywords['tip_'+str(index)]))
    result_sentences =  [embed(line_version)["outputs"] for line_version in line_versions]
    similarity_matrices = [list(np.inner(result_sentence, result_tipificacao)[0]) for result_sentence in result_sentences]
    max_values = [max(similarity_matrice) for similarity_matrice in similarity_matrices]
    max_abs = max(max_values)
    similarity_matrix = similarity_matrices[max_values.index(max_abs)]
    sugestao = tipificacao_tipo[similarity_matrix.index(max_abs)]
    response[index] = (sugestao, str(max_abs))


def getProblem(features):
    response = [None] * 4
    threads = []
    
    threads.append(Thread(target=getFeatureSuggestion, args=(features[0], keywords, result_sintoma, sintoma, response, 0)))
    for i in range(0,3):
        threads.append(Thread(target=getFeatureSuggestion, args=(features[i+1], keywords, emb_og_typifications[i][0], emb_og_typifications[i][1], response, i+1)))

    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()

    return response
