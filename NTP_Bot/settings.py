UPROMPT = [
    'É necessário realizar uma autenticação que mostre que és cliente da NOS.\nIndica o teu nº de telemóvel e o NIF separados por um espaço (exemplo \'930000000 123456789\').',
    'Indica em qual dos seguintes serviços necessitas de assistência: TV, Internet, Voz.',
    'Descreve o problema em que necessitas de ajuda.',
    'Não foi possível reconhecer qual o serviço a que te referes, tenta novamente (TV, Internet ou Voz).',
    'Não foi possível obter dados de um cliente com os dados que inseriste. Envia novamente o teu nº de telemóvel e o NIF separados por um espaço (exemplo \'930000000 123456789\').',
    'Aconteceu um erro ao efetuar a operação de autenticação. Envia novamente o teu nº de telemóvel e o NIF separados por um espaço (exemplo \'930000000 123456789\').',
    'Não foi possível reconhecer o problema descrito, tenta descrever o mesmo novamente.',
    'Foi atingido o máximo de tentativas para esta etapa do processo. A sair do modo de resolução de problemas técnicos...',
    'Não foi possível reconhecer a tua mensagem. A sugestão resolveu o seu problema? (sim/não)',
    'Ainda bem que o seu problema ficou resolvido. A sair do modo de resolução de problemas técnicos...',
    'Não existem mais sugestões para resolver o seu problema, recomendamos que entre em contacto com as linhas de apoio. A sair do modo de resolução de problemas técnicos...',
    'Não tens esse tipo de serviço contratado. Por favor, tenta novamente (TV, Internet ou Voz).'
]
SOLVER_ENDPOINT = 'http://{}:{}/problems/'
SOLVER_ENDPOINT_LOGIN = SOLVER_ENDPOINT + 'login'
SOLVER_ENDPOINT_SOLVE = SOLVER_ENDPOINT + 'solve'
SOLVER_ENDPOINT_SERVICE_CHECK = SOLVER_ENDPOINT + 'client_has_service'
SOLVER_ENDPOINT_UPDATE_LOG = SOLVER_ENDPOINT + 'update_log'
MAX_ERROR_COUNT = 3
FILENAME = 'problems_log.csv'
LOG_MIN_NUMBER_LINES = 15000
REDIS_HOST= ''
REDIS_PORT= 0