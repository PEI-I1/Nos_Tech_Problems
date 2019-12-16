from django.apps import AppConfig
from .sentence_similarity_features import loadModelData


class TechnicalProblemsConfig(AppConfig):
    name = 'technical_problems'

    def ready(self):
        loadModelData()
