from .models import *
from django.shortcuts import render
from django.contrib.auth import authenticate
from django.http import HttpResponse
import json
import os
from .model_prediction import load_model, predict_resolution, load_dict
from .sentence_similarity_features import getProblem
from django.db import IntegrityError

model = load_model(os.getcwd() + '/technical_problems/model_files/model')

def register(request):
    username = request.GET.get('username', '')
    password = request.GET.get('password', '')
    equipamento_tipo = request.GET.get('equipamento_tipo', '')
    tarifario = request.GET.get('tarifario', '')
    if username and password:
        equipamento_tipo_object = Equipamento_Tipo.objects.all().filter(name = equipamento_tipo)
        if len(equipamento_tipo_object) == 0:
            response_as_json = json.dumps({'error': '\'equipamento_tipo\' value is invalid'})
        else:
            equipamento_tipo_object = equipamento_tipo_object[0]
        
            tarifario_object = Tarifario.objects.all().filter(name = tarifario)
            if len(tarifario_object) == 0:
                response_as_json = json.dumps({'error': '\'tarifario\' value is invalid'})
            else:
                tarifario_object = tarifario_object[0]

                user_entry = User(
                    username = username,
                    password = password
                )
                try:
                    user_entry.save()
                    client_entry = Client(
                        user = user_entry,
                        equipamento_tipo = equipamento_tipo_object,
                        tarifario = tarifario_object
                    )
                    client_entry.save()
                    response_as_json = json.dumps({'success': 'User has been registered'})
                except IntegrityError:
                    response_as_json = json.dumps({'error': 'User already exists'})
                    
    else:
        response_as_json = json.dumps({'error': 'Bad parameters'})
    return HttpResponse(response_as_json, content_type='json')


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
        
        if servico in ['TV', 'Internet', 'Voz']:

            clients = Client.objects \
                        .filter(user__username=phone_number, user__password=nif) \
                        .values_list('equipamento_tipo__name', 'tarifario__name')

            if clients:
                client = clients[0]
                equipamento = client[0]
                tarifario = client[1]

                problem = getProblem([sintoma, tipificacao_tipo_1, tipificacao_tipo_2, tipificacao_tipo_3])
                sintoma = problem[0][0]
                tipificacao_tipo_1 = problem[1][0]
                tipificacao_tipo_2 = problem[2][0]
                tipificacao_tipo_3 = problem[3][0]
                print()

                input = [
                    equipamento,
                    servico,
                    problem[0][0], # Sintoma
                    tarifario,
                    problem[1][0], # Tipificação Nivel 1
                    problem[2][0], # Tipificação Nivel 2
                    problem[3][0], # Tipificação Nivel 3
                ]
                
                prediction,probability = predict_resolution(input,model) #'Desliga e volta a ligar', 0.56 

                similarity_features = {
                    'sintoma': {'sugestão': problem[0][0], 'certeza': problem[0][1]},
                    'tipificacao_1': {'sugestão': problem[1][0], 'certeza': problem[1][1]},
                    'tipificacao_2': {'sugestão': problem[2][0], 'certeza': problem[2][1]},
                    'tipificacao_3': {'sugestão': problem[3][0], 'certeza': problem[3][1]},
                }

                response_as_json = json.dumps({'similarity': similarity_features, 'prediction': prediction, 'probability': probability})
            
            else:
                response_as_json = json.dumps({'error': 'Can\'t find client'})

        else:
            response_as_json = json.dumps({'error': 'Parameter \'servico\' has to be \'TV\', \'Internet\' or \'Voz\''})
    
    else:
        response_as_json = json.dumps({'error': 'Bad parameters'})

    return HttpResponse(response_as_json, content_type='json')

