from django.db import models
from django.contrib.auth.models import User

class Client(models.Model):
    """ NOS client
    """
    #username = models.CharField(max_length=256)
    #password = models.CharField(max_length=256)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    equipamento_tipo = models.ForeignKey(
        'Equipamento_Tipo',
        on_delete=models.CASCADE
    )
    tarifario = models.ForeignKey(
        'Tarifario',
        on_delete=models.CASCADE
    )

class Equipamento_Tipo(models.Model):
    name = models.CharField(max_length=256)
    def get_by_natural_key(self, equipamento_tipo):
        return self.get(name=equipamento_tipo)

class Tarifario(models.Model):
    name = models.CharField(max_length=256)
    def get_by_natural_key(self, tarifario):
        return self.get(name=tarifario)
