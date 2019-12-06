from .models import *
from django.shortcuts import render
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.middleware.csrf import get_token
from django.contrib.auth import authenticate, login, logout
from django.conf import settings
import json
import os
from .model_prediction import load_model, predict_resolution, load_dict
from .sentence_similarity_features import getProblem
from django.db import IntegrityError

model = load_model(os.getcwd() + '/technical_problems/model_files/model')

def log_in(request):
    """ Log a user(client) in the system
    """
    #TODO: move logic to dedicated controller
    try:
        uname = request.GET.get('username', '') # phone number
        pwd = request.GET.get('password', '')   # NIF
        if uname and pwd:
            user = authenticate(request, username=uname, password=pwd)
            if user is not None:
                login(request, user)
                return HttpResponse(status=200)
            else:
                return HttpResponse(status=401)
    except:
        return HttpResponse(status=400)

def log_out(request):
    """ Log out a user(client) and clear session data
    """
    logout(request)
    return HttpResponse(status=200)

def register(request):
    uname = request.GET.get('username', '')
    pwd = request.GET.get('password', '')
    equipamento_tipo = request.GET.get('equipamento_tipo', '')
    tarifario = request.GET.get('tarifario', '')
    if uname and pwd:
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

                try:
                    user_entry = User.objects.create_user(uname, '', pwd)
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

# FIXME: uncomment in production
#@login_required
def solve(request):
    #TODO, check that user is authenticated
    if not request.user.is_authenticated:
        return HttpResponse('User not logged in!', status=401)
    else:
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

