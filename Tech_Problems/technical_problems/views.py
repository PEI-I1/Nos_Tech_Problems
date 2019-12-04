from .models import *
from django.shortcuts import render
from django.contrib.auth import authenticate
from django.http import HttpResponse
import json
import os
from .model_prediction import load_model, predict_resolution, load_dict

model = load_model(os.getcwd() + '/technical_problems/model')

def authenticate(request):
    '''
    nif = request.GET.get('nif', '')
    phone_number = request.GET.get('phone_number', '')
    if nif and phone_number:
        user = authenticate(username=phone_number, password=nif)
    else:
        # Devolver resposta que precisa de valores
    '''
    pass

def solve(request):
    #TODO, check that user is authenticated

    phone_number = "933333333"
    nif = "111111111"

    sintoma = request.GET.get('sintoma', '')
    tipificacao_tipo_1 = request.GET.get('tipificacao_tipo_1', '')
    tipificacao_tipo_2 = request.GET.get('tipificacao_tipo_2', '')
    tipificacao_tipo_3 = request.GET.get('tipificacao_tipo_3', '')
    servico = request.GET.get('servico', '')

    if sintoma and tipificacao_tipo_1 and tipificacao_tipo_2 and tipificacao_tipo_3 and servico:

        client = Client.objects \
                       .filter(username=phone_number, password=nif) \
                       .values_list('equipamento_tipo__name', 'tarifario__name')[0]

        equipamento = client[0]
        tarifario = client[1]

        input = [
            equipamento,
            servico,
            sintoma,
            tarifario,
            tipificacao_tipo_1,
            tipificacao_tipo_2,
            tipificacao_tipo_3,
        ]
        
        prediction,probability = predict_resolution(input,model) #'Desliga e volta a ligar', 0.56 

        response_as_json = json.dumps({'prediction': prediction, 'probability': probability, 'equipamento': equipamento, 'tarifario': tarifario})
    
    else:
        response_as_json = json.dumps({'error': 'Bad parameters'})

    return HttpResponse(response_as_json, content_type='json')

