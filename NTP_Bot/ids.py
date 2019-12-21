#!/usr/bin/env python3
import json
import sys
import random

class IDS_State:
    ''' Keeps track of the iterative search state
    for a single client
    '''
    def __init__(self):
        pass


def get_match(prob_dct, search_space):
    #TODO: use sentence similarity to search in search space
    return search_space[0], random.uniform(0,1)


def iter_deepening_search(service, usip):
    ''' Perform an iterative deepening search based
    on user input and the service type
    :param: service type
    :param: user input
    '''
    mia = {
        'Sintoma': ('', 0.0),
        'Tipificacao_Nivel_1':  ('', 0.0),
        'Tipificacao_Nivel_2':  ('', 0.0),
        'Tipificacao_Nivel_3':  ('', 0.0),
    }
    
    with open('tree_options.json', 'r') as stj:
        st = json.load(stj)
            
    cs = st['Servico'][service]
    for ia in mia:
        cs = cs[ia]
        ss = [st for st in cs]
        mt,pb = get_match(usip, ss)
        if pb < 0.65:
            #TODO: request more info OR return support contacts
            break
        mia[ia] = (mt,pb)
        if ia != 'Tipificacao_Nivel_3':
            cs = cs[mt]
            print(mia)
