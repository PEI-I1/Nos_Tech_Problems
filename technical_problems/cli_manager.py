import django.contrib.auth as dAuth
from django.contrib.auth.models import User
from .models import Equipamento_Tipo, Tarifario, Client

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


def register(uname, pwd, equipamento_tipo, tarifario):
    ''' Register a new NOS client
    :param: client username
    :param: client password
    :param: list of equipemnts rented by the client
    :param: mobile tariffs associated with the client
    :return: integer code
    '''
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
    clients = Client.objects \
                    .filter(user__username=uname) \
                    .values_list('equipamento_tipo__name', 'tarifario__name')
    if clients:
        client = clients[0]
        client_info.append(client[0])
        client_info.append(client[1])
        
    return client_info

