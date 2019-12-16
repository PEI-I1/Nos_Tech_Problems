#!/usr/bin/env python3
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import json
import fileinput
import re
from .keywords import keywords
import os
from threading import Thread

embed = None
emb_og_typifications = []
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
    result_sintoma, sintoma = loadProblems('Sintoma')
    for i in range(1,4):
        emb_og_typifications.append(loadProblems('Tipificacao_Nivel_'+str(i)))
    print('-------------> READY <-------------')


def loadProblems(feature_name):
    ''' 
    '''
    with open(os.getcwd() + '/technical_problems/model_files/input_options.json') as json_file:
        data = json.load(json_file)
        feature_og = [x for x in data[feature_name]]
        feature = [x.lower() for x in feature_og]
    return embed(feature)["outputs"], feature_og


def replaceWithKeywords(line, keyword_data):
    keyword_versions = [line]
    for keyword, matches in keyword_data.items():
        keyword_versions.extend([re.sub(match, keyword, line) for match in matches if re.search(match, line)])

    return keyword_versions


def getFeatureSuggestion(line, data, result_tipificacao, tipificacao_tipo, response, index):
    line_versions = replaceWithKeywords(line.lower(), data['common'])
    result_sentences =  [embed(line_version)["outputs"] for line_version in line_versions]
    similarity_matrices = [list(np.inner(result_sentence, result_tipificacao)[0]) for result_sentence in result_sentences]
    max_values = [max(similarity_matrice) for similarity_matrice in similarity_matrices]
    max_abs = max(max_values)
    similarity_matrix = similarity_matrices[max_values.index(max_abs)]
    sugestao = tipificacao_tipo[similarity_matrix.index(max_abs)]
    response[index] = (sugestao, str(max_abs))


def getProblem(features):
    data = keywords
    response = [None] * 4
    threads = []
    
    threads.append(Thread(target=getFeatureSuggestion, args=(features[0], data, result_sintoma, sintoma, response, 0)))
    for i in range(1,4):
        threads.append(Thread(target=getFeatureSuggestion, args=(features[i], data, emb_og_typifications[i][0], emb_og_typifications[i][1], response, i)))

    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()

    return response
