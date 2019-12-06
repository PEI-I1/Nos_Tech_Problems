from .models import *
from django.shortcuts import render
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.middleware.csrf import get_token
from django.conf import settings
import json
import os
from .model_prediction import load_model, predict_resolution, load_dict
from .sentence_similarity_features import getProblem

MODEL = None

def login(request):
    """ Log a user(client) in the system
    """
    #TODO: move logic to dedicated controller
    try:
        uname = request.GET.get('username', '') # phone number
        pwd = request.GET.get('password', '')   # NIF
        if uname and pwd:
            user = User.objects.get(username=uname)
            user = authenticate(request, username=uname, password=pwd)
            if user is not None:
                login(request, user)
                return HttpResponse(status=200)
            else:
                return HttpResponse(status=401)
    except:
        return HttpResponse(status=400)

def logout(request):
    """ Log out a user(client) and clear session data
    """
    logout(request)
    return HttpResponse(status=200)


# FIXME: uncomment in production
#@login_required
def solve(request):
    if not(MODEL): #lazy load
        MODEL = load_model(os.getcwd() + '/technical_problems/model_files/model')

    #TODO, check that user is authenticated
    if not request.user.is_authenticated:
        return HttpResposne('User not logged in!', status=401)
    else:
        sintoma = request.GET.get('sintoma', '')
        tipificacao_tipo_1 = request.GET.get('tipificacao_tipo_1', '')
        tipificacao_tipo_2 = request.GET.get('tipificacao_tipo_2', '')
        tipificacao_tipo_3 = request.GET.get('tipificacao_tipo_3', '')
        servico = request.GET.get('servico', '')

        if sintoma and tipificacao_tipo_1 and tipificacao_tipo_2 and tipificacao_tipo_3 and servico:
            
            if servico in ['TV', 'Internet', 'Voz']:

                clients = Client.objects \
                                .filter(username=phone_number, password=nif) \
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

