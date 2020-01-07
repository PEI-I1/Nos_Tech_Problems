from django.http import HttpResponse
from django.db import IntegrityError
from . import cli_manager as cm
from .solver import predict_resolution, update_models_data
import json, os
from multiprocessing import Process
from .models import Equipamento_Tipo
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt


#FIXME: 
@csrf_exempt
def receive_csv(request):
    ''' Receive log of problems solved on NTP_Bot to improve the model
    '''
    if request.method == 'POST':
        csv = request.FILES['problems_log']
        #FIXME: implement using Celery task and check for result
        p = Process(target=update_models_data, args=(csv,))
        p.start()

        return HttpResponse(status=200)
    else:
        return HttpResponse(status=405)


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
    ''' Register a new user/client in the database to allow
    for technical assistance to be provided
    '''
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


@login_required
def client_has_service(request):
    ''' Check if a client has a specific service in his contract
    '''
    servico = request.GET.get('servico', '')
    uname = request.user.username

    if servico:
        response = cm.cli_has_service(uname, servico)

        response_as_json = json.dumps({
            'has': response
        })
    else:
        response_as_json = json.dumps({
            'has': False
        })

    return HttpResponse(response_as_json, content_type='json')


@login_required
def solve(request):
    ''' Provide a solution to the problem described by the input
    parameters
    '''
    sint = request.GET.get('sintoma', '')
    tip_1 = request.GET.get('tipificacao_tipo_1', '')
    tip_2 = request.GET.get('tipificacao_tipo_2', '')
    tip_3 = request.GET.get('tipificacao_tipo_3', '')
    servico = request.GET.get('servico', '')
    uname = request.user.username
    
    cli_info = cm.get_cli_info(uname, servico)
    
    if cli_info:
        if 'erro' in cli_info:
            if cli_info['erro'] == 1:
                response_as_json = json.dumps({
                    'status': 1,
                    'error': 'Client doesn\'t have a device with that type of service'
                })
            else:
                response_as_json = json.dumps({
                    'status': 2,
                    'error': 'Can\'t find client'
                })
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
                    
            top_resols = predict_resolution(input)

            response_as_json = json.dumps({'status': 0,
                                           'equipamento': cli_info['equipamento'],
                                           'tarifario': cli_info['tarifario'],
                                           'res': [{'prediction': pred, 'probability': prob} for pred,prob in top_resols]
            })
    else:
        response_as_json = json.dumps({
            'status': 3,
            'error': 'Unexpected error'
        })

    return HttpResponse(response_as_json, content_type='json')

