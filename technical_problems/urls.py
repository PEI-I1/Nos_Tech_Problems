from django.urls import path, include
from . import views

urlpatterns = [
    path('authenticate', views.authenticate),
    path('solve', views.solve),
    path('register', views.register)
]