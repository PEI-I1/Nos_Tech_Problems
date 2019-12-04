from django.db import models
from django.contrib.auth.models import AbstractUser

class Client(AbstractUser):
    """ NOS client
    """
    equipamento_tipo = models.ForeignKey(
        'Equipamento_Tipo',
        on_delete=models.CASCADE
    )
    tarifario = models.ForeignKey(
        'Tarifario',
        on_delete=models.CASCADE
    )
    tecnologia = models.ForeignKey(
        'Tecnologia',
        on_delete=models.CASCADE
    )

class Equipamento_Tipo(models.model):
    name = models.CharField(max_length=256)

class Tarifario(models.model):
    name = models.CharField(max_length=256)

class Tecnologia(models.model):
    name = models.CharField(max_length=256)

