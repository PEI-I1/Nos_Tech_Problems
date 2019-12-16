from django.http import HttpResponse
from django.db import IntegrityError
from .models import ServiceType
from .sentence_similarity_features import getProblem
from . import cli_manager as cm
from .import model_prediction as mp
import json, os

model = mp.load_model(os.getcwd() + '/technical_problems/model_files/model')

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
    equipamento_tipo = request.GET.get('equipamento_tipo', '')
    tarifario = request.GET.get('tarifario', '')
    rc = cm.register(uname, pwd, equipamento_tipo, tarifario)
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
    tipificacao_tipo_1 = request.GET.get('tipificacao_tipo_1', '')
    tipificacao_tipo_2 = request.GET.get('tipificacao_tipo_2', '')
    tipificacao_tipo_3 = request.GET.get('tipificacao_tipo_3', '')
    servico = request.GET.get('servico', '')
    uname = request.user.getUsername()
    
    cli_info = get_cli_info(uname)
    
    if len(cli_info) > 0:
        if sintoma and tip_1 and tip_2 and tip_3 and serv:
            services_hr = [service[1] for service in models.ServiceType.AVAILABLE_SERVICES]
            if serv in services_hr:
                sint, tip_1, tip_2 , tip_3 = getProblem([sintoma, tip_1, tip_2, tip_3])
                
                input = [
                    equipamento,
                    servico,
                    sint[0], # Sintoma
                    tarifario,
                    tip_1[0], # Tipificação Nivel 1
                    tip_2[0], # Tipificação Nivel 2
                    tip_3[0], # Tipificação Nivel 3
                ]
                
                prediction,probability = mp.predict_resolution(input, model) #'Desliga e volta a ligar', 0.56 

                similarity_features = {
                    'sintoma': {'sugestão': sint[0], 'certeza': sint[1]},
                    'tipificacao_1': {'sugestão': tip_1[0], 'certeza': tip_1[1]},
                    'tipificacao_2': {'sugestão': tip_2[0], 'certeza': tip_2[1]},
                    'tipificacao_3': {'sugestão': tip_3[0], 'certeza': tip_3[1]},
                }

                response_as_json = json.dumps({'similarity': similarity_features, 'prediction': prediction, 'probability': probability})
            else:
                response_as_json = json.dumps({'error': 'Parameter \'servico\' has to be \'TV\', \'Internet\' or \'Voz\''})
        else:
            response_as_json = json.dumps({'error': 'Bad parameters'})            
    else:
        response_as_json = json.dumps({'error': 'Can\'t find client'})

    return HttpResponse(response_as_json, content_type='json')
