#!/usr/bin/env python3
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import json
import fileinput
import re
import keywords

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2")

def loadProblems():
    with open('input_options.json') as json_file:
        data = json.load(json_file)
        tipificacao_tipo = [x.lower() for x in data['Tipificacao_Nivel_2']]
    return embed(tipificacao_tipo)["outputs"], tipificacao_tipo



def replaceWithKeywords(line, keyword_data):
    keyword_versions = [line]
    for keyword, matches in keyword_data.items():
        keyword_versions.extend([re.sub(match, keyword, line) for match in matches if re.search(match, line)])

    print(keyword_versions)
    return keyword_versions
    

def getProblem(result_tipificacao, tipificacao_tipo):
    data = keywords.keywords
    print("Introduza o seu problema técnico:")
    for line in fileinput.input():
        line_versions = replaceWithKeywords(line.lower(), data)
        result_sentences =  [embed(line_version)["outputs"] for line_version in line_versions]
        similarity_matrices = [list(np.inner(result_sentence, result_tipificacao)[0]) for result_sentence in result_sentences]
        max_values = [max(similarity_matrice) for similarity_matrice in similarity_matrices]
        max_abs = max(max_values)
        similarity_matrix = similarity_matrices[max_values.index(max_abs)]
        print('Tipificação sugerida: ' + tipificacao_tipo[similarity_matrix.index(max_abs)] + ', valor = ' + str(max_abs))


if __name__ == "__main__":
    result_tipificacao, tipificacao_tipo = loadProblems()
    getProblem(result_tipificacao, tipificacao_tipo)
