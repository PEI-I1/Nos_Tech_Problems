from django.http import HttpResponse
from django.db import IntegrityError
from . import cli_manager as cm
from .solver import load_model, predict_resolution
import json, os
from .models import Equipamento_Tipo
from django.contrib.auth.decorators import login_required

model = load_model(os.getcwd() + '/technical_problems/model_files/model')

def login(request):
    ''' Log a user(client) in the system
    '''
    uname = request.GET.get('username', '') # phone number
    pwd = request.GET.get('password', '')   # NIF
    login = cm.login(request, uname, pwd)
    if login == 0:
        return HttpResponse(status=200) # Success
    elif login == 1:
        return HttpResponse(status=401) # Unauthorized
    else:
        return HttpResponse(status=500) # Internal Error

    
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
@login_required
def solve(request):
    sint = request.GET.get('sintoma', '')
    tip_1 = request.GET.get('tipificacao_tipo_1', '')
    tip_2 = request.GET.get('tipificacao_tipo_2', '')
    tip_3 = request.GET.get('tipificacao_tipo_3', '')
    servico = request.GET.get('servico', '')
    #uname = request.GET.get('username', '') #FIXME: remove in production
    uname = request.user.username
    print('username = ' + uname)
    
    cli_info = cm.get_cli_info(uname, servico)
    
    if cli_info:
        if 'erro' in cli_info:
            if cli_info['erro'] == 1:
                response_as_json = json.dumps({'error': 'Client doesn\'t have a device with that type of service'})
            else:
                response_as_json = json.dumps({'error': 'Can\'t find client'})
        else:
            input = [
                cli_info['equipamento'],
                cli_info['servico'],
                sint,
                cli_info['tarifario'],
                tip_1,
                tip_2,
                tip_3,
            ]
                    
            prediction,probability = predict_resolution(input, model) #'Desliga e volta a ligar', 0.56 

            response_as_json = json.dumps({'status': 0,
                                           'res': {
                                               'prediction': prediction,
                                               'probability': probability
                                           }})                    
    else:
        response_as_json = json.dumps({'error': 'Unexpected error'})

    print(response_as_json)
    return HttpResponse(response_as_json, content_type='json')
