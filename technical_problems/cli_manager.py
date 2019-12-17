import django.contrib.auth as dAuth
from django.contrib.auth.models import User
from .models import Contrato, Cliente, Equipamento_Tipo, Tarifario
import json
from django.db import IntegrityError

def login(request, uname, pwd):
    ''' Log a user in and create an HTTP session
    '''
    try:
        if uname and pwd:
            user = dAuth.authenticate(request, username=uname, password=pwd)
            if user is not None:
                dAuth.login(request, user)
                return 0
            else:
                return 1
    except:
        return 2


def logout(request):
    ''' Log a user out and clean session data
    :param: HTTP request with session data
    '''
    dAuth.logout(request)


def register(uname, pwd, morada, equipamentos_tipo, tarifario):
    ''' Register a new NOS client
    :param: client username
    :param: client password
    :param: contract address
    :param: list of equipemnts rented by the client
    :param: mobile tariffs associated with the client
    :return: integer code
    '''
    if uname and pwd and morada and equipamentos_tipo and tarifario:
        equipamentos = equipamentos_tipo.split(',')
        equipamentos_tipo_object = Equipamento_Tipo.objects.all().filter(nome__in = equipamentos)
        print(equipamentos_tipo_object)
        if len(equipamentos_tipo_object) != len(equipamentos):
            response_as_json = json.dumps({'error': '\'equipamentos\' values are invalid'})
        else:
            tarifario_object = Tarifario.objects.all().filter(nome = tarifario)
            if len(tarifario_object) == 0:
                response_as_json = json.dumps({'error': '\'tarifario\' value is invalid'})
            else:
                tarifario_object = tarifario_object[0]
                try:
                    contract_entry = Contrato(
                        morada  = morada,
                        tarifario = tarifario_object
                    )
                    contract_entry.save()
                    contract_entry.equipamentos.set(equipamentos_tipo_object)
                    user_entry = User.objects.create_user(uname, '', pwd)
                    client_entry = Cliente(
                        user = user_entry,
                        contrato = contract_entry
                    )
                    client_entry.save()
                    return 0
                except IntegrityError:
                    return 1
    else:
        return 2


def get_cli_info(uname):
    ''' Fetch information regarding the given client
    :param: client username
    :return: list with Equipamento_Tipo and Tarifario
    '''
    client_info = []
    address = Cliente.objects \
                     .filter(user__username=uname) \
                     .values_list('contrato')
    if clients:
        contract_info = Contrato.objects \
                                .filter(morada=address) \
                                .values_list('tarifario', 'equipamentos', flat=True)
        client_info.append(contract_info[0])
        client_info.append(contract_info[1])  # FIXME
        service_type_hr = Equipamento_Tipo.objects.get(pk=contract_info[1]).get_servico_display()
        client_info.append(service_type_hr)
        
    return client_info

