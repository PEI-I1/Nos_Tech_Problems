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
    model_args: dictionary
    Maps each input argument to it's value and confidence (based on semantic similarity)
    '''
    def __init__(self):
        self.state = 0
        self.session = None
        self.service = ''
        self.error_count = 0
        self.model_args = {
            'Sintoma': ('', 0.0),
            'Tipificacao_Nivel_1':  ('', 0.0),
            'Tipificacao_Nivel_2':  ('', 0.0),
            'Tipificacao_Nivel_3':  ('', 0.0)
        }
        self.equipment = ''
        self.tariff = ''
        self.suggestions = ''
        self.suggestion = ''
        self.suggestion_count = 0

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

    def get_problem_solution(self):
        ''' Use solver backend to retrieve a possible solution for the problem
        :return: possible solution
        '''

        sintoma = self.model_args['Sintoma'][0]
        tip_1 = self.model_args['Tipificacao_Nivel_1'][0]
        tip_2 = self.model_args['Tipificacao_Nivel_2'][0]
        tip_3 = self.model_args['Tipificacao_Nivel_3'][0]

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
        self.equipment = solver_json['equipamento']
        self.tariff = solver_json['tarifario']
        self.suggestions = solver_json['res']
        print(self.suggestions)
        if status == 0: # Success
            self.suggestion = self.suggestions[self.suggestion_count]['prediction']
            return 1, self.suggestion
        elif status == 1:
            return 0, 'Não existe informação de algum equipamento seu com serviço de ' + self.service + '.'
        elif status == 2:
            return 0, 'Não foi possível encontrar os seus detalhes de cliente. Tente de novo.'
        else:
            return 0, 'Ocorreu um erro durante a operação. Tente de novo.'

    def iter_deepening_search(self, prob_desc):
        ''' Perform an iterative deepening search based
        on user input and the service type
        :param: user input
        :return: True on success, else False
        '''
        with open('tree_options.json', 'r') as search_tree_json:
            search_tree = json.load(search_tree_json)
        
        cs = search_tree['Servico'][self.service]
        for input_arg in self.model_args:
            cs = cs[input_arg]
            search_space = [search_tree for search_tree in cs]
            mt, prob = msg_interpreter.extractProblemData(prob_desc, search_space, 0)
            if prob < 0.65: #FIXME: change threshold to 0.65
                #print(self.model_args)
                return False
            elif prob > self.model_args[input_arg][1]:
                self.model_args[input_arg] = (mt, prob)

            if input_arg != 'Tipificacao_Nivel_3':
                cs = cs[mt]

        #print(self.model_args)
        return True


    def check_client_services(self, service): 
        ''' Check if a client has a specific service in his contract
        :param: service to check
        :return: True or False
        '''
        checker = self.session.get(
            settings.SOLVER_ENDPOINT_SERVICE_CHECK,
            params={
                'servico': service
            }
        )
        checker_json = json.loads(checker.text)
        #print(checker_json)
        res = checker_json['has']
        return res