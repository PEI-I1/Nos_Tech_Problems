from django.urls import path, include
from . import views

urlpatterns = [
    path('login', views.login),
    path('logout', views.logout),
    path('solve', views.solve),
    path('register', views.register)
]
