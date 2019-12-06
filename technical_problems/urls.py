from django.urls import path, include
from . import views

urlpatterns = [
    path('login', views.log_in),
    path('logout', views.log_out),
    path('solve', views.solve),
    path('register', views.register)
]
