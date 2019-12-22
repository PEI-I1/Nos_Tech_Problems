from django.http import HttpResponse
from django.db import IntegrityError
from .speech_interpreter import getProblem
from . import cli_manager as cm
from .solver import load_model, predict_resolution
import json, os
from .models import Equipamento_Tipo

model = load_model(os.getcwd() + '/technical_problems/model_files/model')

def login(request):
    ''' Log a user(client) in the system
    '''
    uname = request.GET.get('username', '') # phone number
    pwd = request.GET.get('password', '')   # NIF
    cli_auth.login(request, uname, pwd)

    
def logout(request):
    ''' Log out a user(client) and clear session data
    '''
    rc = cm.login(request)
    if not(rc):
        return HttpResponse(status=200)
    elif rc == 1:
        return HttpResponse(status=401)
    else:
        return HttpResponse(status=400)


def register(request):
    uname = request.GET.get('username', '')
    pwd = request.GET.get('password', '')
    morada = request.GET.get('morada', '')
    equipamentos_tipo = request.GET.get('equipamentos', '')
    tarifario = request.GET.get('tarifario', '')
    rc = cm.register(uname, pwd, morada, equipamentos_tipo, tarifario)
    if not(rc): # success
        response_as_json = json.dumps({'success': 'User has been registered'})
    elif rc == 1: # duplicate user
        response_as_json = json.dumps({'error': 'User already exists'})
    else:
        response_as_json = json.dumps({'error': 'Bad parameters'})
    return HttpResponse(response_as_json, content_type='json')


# FIXME: uncomment in production
#@login_required
def solve(request):
    sintoma = request.GET.get('sintoma', '')
    tip_1 = request.GET.get('tipificacao_tipo_1', '')
    tip_2 = request.GET.get('tipificacao_tipo_2', '')
    tip_3 = request.GET.get('tipificacao_tipo_3', '')
    servico = request.GET.get('servico', '')
    uname = request.GET.get('username', '') #FIXME: remove in production
    #uname = request.user.getUsername()
    
    cli_info = cm.get_cli_info(uname, servico)
    
    if len(cli_info) > 0:
        sint, tip_1, tip_2 , tip_3 = getProblem([sintoma, tip_1, tip_2, tip_3])

        if equipamento:

            input = [
                cli_info['equipamento'],
                servico,
                sint[0],
                cli_info['tarifario'],
                tip_1[0],
                tip_2[0],
                tip_3[0],
            ]
                    
            prediction,probability = predict_resolution(input, model) #'Desliga e volta a ligar', 0.56 

            similarity_features = {
                'sintoma': {'sugestão': sint[0], 'certeza': sint[1]},
                'tipificacao_1': {'sugestão': tip_1[0], 'certeza': tip_1[1]},
                'tipificacao_2': {'sugestão': tip_2[0], 'certeza': tip_2[1]},
                'tipificacao_3': {'sugestão': tip_3[0], 'certeza': tip_3[1]},
            }

            response_as_json = json.dumps({'similarity': similarity_features, 'prediction': prediction, 'probability': probability})
                    
        else:
            response_as_json = json.dumps({'error': 'Client doesn\'t have a device with that type of service'})
    else:
        response_as_json = json.dumps({'error': 'Can\'t find client'})

    return HttpResponse(response_as_json, content_type='json')
