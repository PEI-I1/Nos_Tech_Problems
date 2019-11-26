import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import json
import fileinput

#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/2")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2")

'''
# Some texts of different lengths.
technical_sentences_1 = ["Rede sem fios lenta", "wifi não funciona", "Baixo sinal de wifi"]
technical_sentences_2 = ["LENTIDAO ACESSO WIRELESS"]

# Compute embeddings.
result_1 = embed(technical_sentences_1)["outputs"]
result_2 = embed(technical_sentences_2)["outputs"]

# Compute similarity matrix. Higher score indicates greater similarity.
similarity_matrix = np.inner(result_1, result_2)
print(similarity_matrix[0])
'''

with open('input_options.json') as json_file:
    data = json.load(json_file)
    tipificacao_tipo = [x.lower() for x in data['Tipificacao_Nivel_2']]
    result_tipificacao = embed(tipificacao_tipo)["outputs"]

print("Introduza o seu problema técnico:")
for line in fileinput.input():
    technical_sentence = line
    result_sentence =  embed(technical_sentence)["outputs"]
    similarity_matrix = list(np.inner(result_sentence, result_tipificacao)[0])
    print('Tipificação sugerida: ' + tipificacao_tipo[similarity_matrix.index(max(similarity_matrix))] + ', valor = ' + str(max(similarity_matrix)))
