UPROMPT = [
    'Por favor indique o seu nº de telemóvel seguido do NIF.',
    'Indique em qual dos seguintes serviços necessita de assistência: TV, Internet, Voz.',
    'Descreva o problema em causa.',
    'Não foi possível reconhecer qual o serviço a que se refere, tente novamente.',
    'Não foi possível obter dados de um cliente com os dados que inseriu. Envie novamente o seu nº de telemóvel e o NIF separados por um espaço.',
    'Aconteceu um erro ao efetuar a operação de autenticação. Envie novamente o seu nº de telemóvel e o NIF separados por um espaço.',
    'Não foi possível reconhecer o problema descrito, tente novamente.',
    'Foi atingido o máximo de tentativas para esta etapa do processo. A sair do modo de resolução de problemas técnicos...',
    'Não foi possível reconhecer a sua mensagem. A sugestão resolveu o seu problema? (sim/não)',
    'Ainda bem que o seu problema ficou resolvido. A sair do modo de resolução de problemas técnicos...',
    'Não existem mais sugestões para resolver o seu problema. A sair do modo de resolução de problemas técnicos...',
    'Não tem esse tipo de serviço contratado. Por favor tente novamente.'
]

SOLVER_ENDPOINT = 'http://127.0.0.1:8000/problems/'
SOLVER_ENDPOINT_LOGIN = SOLVER_ENDPOINT + 'login'
SOLVER_ENDPOINT_SOLVE = SOLVER_ENDPOINT + 'solve'
SOLVER_ENDPOINT_SERVICE_CHECK = SOLVER_ENDPOINT + 'client_has_service'
SOLVER_ENDPOINT_UPDATE_LOG = SOLVER_ENDPOINT + 'update_log'
MAX_ERROR_COUNT = 3
FILENAME = 'problems_log.csv'
LOG_MIN_NUMBER_LINES = 10000
