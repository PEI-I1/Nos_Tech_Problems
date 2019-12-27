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
        try:
            auth_res = self.session.get(
                settings.SOLVER_ENDPOINT_LOGIN,
                params={
                    'username': uname,
                    'password': pwd
                }
            )
            if auth_res.status_code == 200:
                return 0
            elif auth_res.status_code == 401:
                return 1
            else: # auth_res.status_code == 500
                return 2
        except:
            return 2

    def get_problem_solution(self, model_input_args):
        ''' Use solver backend to retrieve a possible solution for the problem
        :param: model input arguments
        :return: possible solution
        '''

        sintoma = model_input_args['Sintoma'][0]
        tip_1 = model_input_args['Tipificacao_Nivel_1'][0]
        tip_2 = model_input_args['Tipificacao_Nivel_2'][0]
        tip_3 = model_input_args['Tipificacao_Nivel_3'][0]

        solver = self.session.get(
            settings.SOLVER_ENDPOINT_SOLVE,
            params={
                'sintoma': sintoma,
                'tipificacao_tipo_1': tip_1,
                'tipificacao_tipo_2': tip_2,
                'tipificacao_tipo_3': tip_3,
                'servico': self.service
            }
        )

        solver_json = json.loads(solver.text)
        print(solver_json)

        status = solver_json['status']
        if status == 0: # Success
            return 1, solver_json['res']['prediction']
        elif status == 1:
            return 0, 'Não existe informação de algum equipamento seu com serviço de ' + self.service + '.'
        elif status == 2:
            return 0, 'Não foi possível encontrar os seus detalhes de cliente. Tente de novo.'
        else:
            return 0, 'Ocorreu um erro durante a operação. Tente de novo.'

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
        #if prob < 0.65:
            #TODO: request more info OR return support contacts
        #    print()
        model_input_args[input_arg] = (mt, prob)
        if input_arg != 'Tipificacao_Nivel_3':
            cs = cs[mt]

    print(model_input_args)
    return model_input_args