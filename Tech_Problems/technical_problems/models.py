from django.db import models
from django.contrib.auth.models import User

class Equipamento_Tipo(models.Model):
    ''' Device used to provide a service
    '''
    nome = models.CharField(max_length=256, default="", unique=True)
    

class Tarifario(models.Model):
    ''' Specific type of service offering
    '''
    nome = models.CharField(max_length=256, default="", unique=True)


class Servico(models.Model):
    ''' Service offering provided to a specific client
    '''
    AVAILABLE_SERVICES = [
        ('tv', 'TV'),
        ('internet', 'Internet'),
        ('voz', 'Voz')
    ]
    servico = models.CharField(
        max_length=8,
        choices=AVAILABLE_SERVICES,
        default='tv'
    )
    tarifario = models.ForeignKey(
        'Tarifario',
        on_delete=models.DO_NOTHING
    )
    equipamento = models.ForeignKey(
        'Equipamento_Tipo',
        on_delete=models.DO_NOTHING
    )


class Contrato(models.Model):
    ''' ISP contract associated with a specific address
    '''
    morada = models.CharField(primary_key=True, max_length=256, default="")
    servicos = models.ManyToManyField(Servico)

    
class Cliente(models.Model):
    ''' ISP client
    '''
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # contracts = models.ManyToManyField(Contract) - support more than one contract
    contrato = models.ForeignKey(
        'Contrato',
        on_delete=models.CASCADE,
    )
