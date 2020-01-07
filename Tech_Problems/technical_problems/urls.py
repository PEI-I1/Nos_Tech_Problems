from django.urls import path, include
from . import views

urlpatterns = [
    path('login', views.login),
    path('logout', views.logout),
    path('solve', views.solve),
    path('register', views.register),
    path('client_has_service', views.client_has_service),
    path('update_log', views.receive_csv)
]
