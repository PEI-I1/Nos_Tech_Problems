#!/usr/bin/env python3
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import json
import fileinput
import re
from .keywords import keywords
import os


#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2")

def loadProblems(feature_name):
    with open(os.getcwd() + '/technical_problems/model_files/input_options.json') as json_file:
        data = json.load(json_file)
        feature_og = [x for x in data[feature_name]]
        feature = [x.lower() for x in feature_og]
    return '', ''
    #return embed(feature)["outputs"], feature_og


result_sintoma, sintoma = loadProblems('Sintoma')
result_tipificacao_1, tipificacao_tipo_1 = loadProblems('Tipificacao_Nivel_1')
result_tipificacao_2, tipificacao_tipo_2 = loadProblems('Tipificacao_Nivel_2')
result_tipificacao_3, tipificacao_tipo_3 = loadProblems('Tipificacao_Nivel_3')
print('-------------> READY <-------------')

def replaceWithKeywords(line, keyword_data):
    keyword_versions = [line]
    for keyword, matches in keyword_data.items():
        keyword_versions.extend([re.sub(match, keyword, line) for match in matches if re.search(match, line)])

    #print(keyword_versions)
    return keyword_versions
    

def getProblem(features):
    data = keywords
    response = []
    i = 0
    for line in features:
        if i == 0:
            result_tipificacao = result_sintoma
            tipificacao_tipo = sintoma
        elif i == 1:
            result_tipificacao = result_tipificacao_1
            tipificacao_tipo = tipificacao_tipo_1
        elif i == 2:
            result_tipificacao = result_tipificacao_2
            tipificacao_tipo = tipificacao_tipo_2
        else:
            result_tipificacao = result_tipificacao_3
            tipificacao_tipo = tipificacao_tipo_3
        line_versions = replaceWithKeywords(line.lower(), data)
        result_sentences =  []#[embed(line_version)["outputs"] for line_version in line_versions]
        similarity_matrices = [list(np.inner(result_sentence, result_tipificacao)[0]) for result_sentence in result_sentences]
        max_values = [max(similarity_matrice) for similarity_matrice in similarity_matrices]
        max_abs = max(max_values)
        similarity_matrix = similarity_matrices[max_values.index(max_abs)]
        sugestao = tipificacao_tipo[similarity_matrix.index(max_abs)]
        response.append((sugestao, str(max_abs)))
        i += 1
    return response
    #print(response)


'''
print("Introduza o seu problema técnico:")
for line in fileinput.input():
    features = line.split(',')
    getProblem(features)
'''
