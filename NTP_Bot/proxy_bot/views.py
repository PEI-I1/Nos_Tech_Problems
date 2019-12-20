from django.http import HttpResponse
from django.shortcuts import render
import json

# Create your views here.

def solve(request):
    response_as_json = json.dumps({'msg': 'Teste'})
    return HttpResponse(response_as_json, content_type='json')