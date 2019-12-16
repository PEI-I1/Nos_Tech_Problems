from django.db import models
from django.contrib.auth.models import User

class Equipamento_Tipo(models.Model):
    name = models.CharField(max_length=256)
    def get_by_natural_key(self, equipamento_tipo):
        return self.get(name=equipamento_tipo)

class Tarifario(models.Model):
    name = models.CharField(max_length=256)
    def get_by_natural_key(self, tarifario):
        return self.get(name=tarifario)

class ServiceType(models.Model):
    AVAILABLE_SERVICES = [
        (1,  'TV'),
        (2, 'Internet'),
        (3, 'Voz')
    ]
    name = models.IntegerField(
        choices=AVAILABLE_SERVICES,
        default=1
    )
    def get_by_natural_key(self, service):
        return self.get(name=service)

class Client(models.Model):
    """ NOS client
    """
    #username = models.CharField(max_length=256)
    #password = models.CharField(max_length=256)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    address = models.CharField(max_length=256)
    services = models.ManyToManyField(ServiceType)
    devices = models.ManyToManyField(Equipamento_Tipo)
    tariffs = models.ManyToManyField(Tarifario)
