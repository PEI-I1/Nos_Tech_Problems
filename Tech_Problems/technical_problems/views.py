from django.shortcuts import render
from django.contrib.auth import authenticate

# Create your views here.

def authenticate(request):
    nif = request.GET.get('nif', '')
    phone_number = request.GET.get('phone_number', '')
    if nif and phone_number:
        user = authenticate(username=phone_number, password=nif)
    else:
        # Devolver resposta que precisa de valores