from django.db import models
from django.contrib.auth.models import User

class Equipamento_Tipo(models.Model):
    name = models.CharField(max_length=256)
    AVAILABLE_SERVICES = [
        (1, 'TV'),
        (2, 'Internet'),
        (3, 'Voz')
    ]
    service = models.IntegerField(
        choices=AVAILABLE_SERVICES,
        default=1
    )
    def get_by_natural_key(self, equipamento_tipo):
        return self.get(name=equipamento_tipo)


class Tarifario(models.Model):
    name = models.CharField(max_length=256)
    def get_by_natural_key(self, tarifario):
        return self.get(name=tarifario)


class Contract(models.Model):
    ''' ISP contract associated with a specific address
    '''
    address = models.CharField(primary_key=True, max_length=256)
    tariffs = models.ForeignKey(
        'Tarifario',
        on_delete=models.DO_NOTHING
    )
    devices = models.ManyToManyField(Equipamento_Tipo)

    
class Client(models.Model):
    ''' NOS client
    '''
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # contracts = models.ManyToManyField(Contract) - support more than one contract
    contract = models.ForeignKey(
        'Contract',
        on_delete=models.CASCADE,
    )
