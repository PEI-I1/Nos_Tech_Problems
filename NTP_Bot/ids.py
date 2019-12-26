#!/usr/bin/env python3
import json
import sys
import random
import msg_interpreter

class IDS_State:
    ''' Keeps track of the iterative search state
    for a single client
    '''
    def __init__(self):
        pass

def iter_deepening_search(prob_desc, service):
    ''' Perform an iterative deepening search based
    on user input and the service type
    :param: user input
    :param: service type
    '''
    model_input_args = {
        'Sintoma': ('', 0.0),
        'Tipificacao_Nivel_1':  ('', 0.0),
        'Tipificacao_Nivel_2':  ('', 0.0),
        'Tipificacao_Nivel_3':  ('', 0.0),
    }
    
    with open('tree_options.json', 'r') as search_tree_json:
        search_tree = json.load(search_tree_json)
            
    cs = search_tree['Servico'][service]
    for input_arg in model_input_args:
        cs = cs[input_arg]
        search_space = [search_tree for search_tree in cs]
        mt, prob = msg_interpreter.extractProblemData(prob_desc, search_space, 0) #FIXME
        if prob < 0.65:
            #TODO: request more info OR return support contacts
            break
        model_input_args[input_arg] = (mt, prob)
        if input_arg != 'Tipificacao_Nivel_3':
            cs = cs[mt]
        print(model_input_args)
