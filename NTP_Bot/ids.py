#!/usr/bin/env python3
import sys, random, json, requests
import settings, msg_interpreter


class Inter_State:
    ''' Keeps track of the state of the client-system interaction
    for a single user
    
    Attributes
    ----------
    state: int
    identifier for the current interaction state
    session: Session
    HTTP session for a successfully authenticated user
    service: str
    Identifier of the service in which the user needs assistance
    '''
    def __init__(self):
        self.state = 0
        self.session = None
        self.service = ''

    def setupSession(self, uname, pwd):
        ''' Sets up an HTTP session with the solver backend
        :param: username
        :param: password
        :return: True if authentication successful else False
        '''
        self.session = requests.Session()
        auth_endpoint = settings.SOLVER_ENDPOINT + 'login'
        try:
            auth_res = self.session.get(auth_endpoint, params={'username':uname, 'pasword':pwd})
            # FIXME: return meaningful error message (e.g. user not found, incorrect username/password)
            return auth_res.status_code == 200
        except:
            return False

def iter_deepening_search(prob_desc, service):
    ''' Perform an iterative deepening search based
    on user input and the service type
    :param: user input
    :param: service type
    :return: input values for solver model
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

    return model_input_args
