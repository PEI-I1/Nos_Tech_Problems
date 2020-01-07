import django.contrib.auth as dAuth
from django.contrib.auth.models import User
from .models import Contrato, Cliente, Equipamento_Tipo, Tarifario, Servico
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

        if len(equipamentos_tipo_object) != len(equipamentos): # invalid equipments provided
            return 2
        else:
            tarifario_object = Tarifario.objects.all().filter(nome = tarifario)
            if len(tarifario_object) == 0:
                return 2
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


def cli_has_service(uname, service):
    ''' Check if a client has a specific service in his contract
    :param: client username
    :param: service to check
    :return: True or False
    '''
    contract = Cliente.objects \
                      .all() \
                      .filter(user__username=uname) \
                      .values_list('contrato', flat=True)

    if contract:
        address = contract[0]

        services = Contrato.objects \
                           .get(morada=address) \
                           .servicos \
                           .all() \
                           .values_list('servico', flat=True)

        if not services:
            return False

        else:
            if service.lower() in services:
                return True
            else:
                return False

    else:
        return False


def get_cli_info(uname, service):
    ''' Fetch information regarding a contracted service
    :param: client username
    :param: service to fetch info about
    :return: list with Equipamento_Tipo, Tarifario and Servico
    '''
    client_info = {}
    contract = Cliente.objects \
                      .all() \
                      .filter(user__username=uname) \
                      .values_list('contrato', flat=True)

    if contract:
        address = contract[0]

        services = Contrato.objects \
                           .get(morada=address) \
                           .servicos \
                           .all()

        service_info = services.filter(servico=service.lower()) \
                               .values_list('tarifario__nome', 'equipamento__nome', 'servico')

        if not service_info:
            client_info['erro'] = 1

        else:
            client_info['tarifario'] = service_info[0][0]
            client_info['equipamento'] = service_info[0][1]
            client_info['servico'] = dict(Servico.AVAILABLE_SERVICES)[service_info[0][2]]
        
    else:
        client_info['erro'] = 2
    
    print(client_info)
    return client_info


