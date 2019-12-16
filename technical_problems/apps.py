from django.apps import AppConfig
from .speech_interpreter import loadModelData


class TechnicalProblemsConfig(AppConfig):
    name = 'technical_problems'

    def ready(self):
        loadModelData()
