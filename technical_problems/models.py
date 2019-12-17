from django.db import models
from django.contrib.auth.models import User

class Equipamento_Tipo(models.Model):
    nome = models.CharField(primary_key=True, max_length=256, default="")
    AVAILABLE_SERVICES = [
        (1, 'TV'),
        (2, 'Internet'),
        (3, 'Voz')
    ]
    servico = models.IntegerField(
        choices=AVAILABLE_SERVICES,
        default=1
    )
    def get_by_natural_key(self, equipamento_tipo):
        return self.get(name=equipamento_tipo)


class Tarifario(models.Model):
    nome = models.CharField(primary_key=True, max_length=256, default="")
    def get_by_natural_key(self, tarifario):
        return self.get(name=tarifario)


class Contrato(models.Model):
    ''' ISP contract associated with a specific address
    '''
    morada = models.CharField(primary_key=True, max_length=256, default="")
    tarifario = models.ForeignKey(
        'Tarifario',
        on_delete=models.DO_NOTHING
    )
    equipamentos = models.ManyToManyField(Equipamento_Tipo)

    
class Cliente(models.Model):
    ''' NOS client
    '''
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # contracts = models.ManyToManyField(Contract) - support more than one contract
    contrato = models.ForeignKey(
        'Contrato',
        on_delete=models.CASCADE,
    )
