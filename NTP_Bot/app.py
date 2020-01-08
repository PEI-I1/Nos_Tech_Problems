import base64, json, pickle, os, redis
import settings
import msg_interpreter
import ids
import requests
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

app = Flask(__name__)

@app.route('/solver', methods=['POST'])
def solve():
    data = request.get_json()
    chat_id = data['idChat']
    msg = data['msg']
    
    ret_dict = {'chat_id': chat_id, 'msg': ''}
    csr = redis_db.get(chat_id)

    save_on_redis = True

    if csr:
        cs = json.loads(csr.decode('utf-8'))
        exec_state = pickle.loads(base64.decodebytes(cs['state'].encode()))
        
        if exec_state.state == 0:
            auth_data = msg.split(' ')
            if len(auth_data) >= 2:
                login = exec_state.setupSession(auth_data[0], auth_data[1])
                if login == 0:
                    ret_dict['msg'] = settings.UPROMPT[1]
                    exec_state.state = exec_state.state + 1
                    exec_state.error_count = 0
                elif login == 1:
                    exec_state.error_count = exec_state.error_count + 1
                    if exec_state.error_count == settings.MAX_ERROR_COUNT:
                        ret_dict['msg'] = settings.UPROMPT[7]
                        ret_dict['chat_id'] = -1
                        save_on_redis = False
                    else:
                        ret_dict['msg'] = settings.UPROMPT[4]
                else:
                    ret_dict['msg'] = settings.UPROMPT[5]
            else:
                ret_dict['msg'] = settings.UPROMPT[5]
                
        elif exec_state.state == 1:
            services_array = [('tv', 'TV'), ('televisão', 'TV'), ('televisao', 'TV'), ('internet', 'Internet'), ('net', 'Internet'), ('wifi', 'Internet'), ('voz', 'Voz')]
            match_service = [x[1] for x in services_array if msg.lower() == x[0]]
            if match_service:
                checker = exec_state.check_client_services(match_service[0])
                if checker:
                    exec_state.service = match_service[0]
                    exec_state.state = exec_state.state + 1
                    exec_state.error_count = 0
                    ret_dict['msg'] = settings.UPROMPT[2]
                else:
                    exec_state.error_count = exec_state.error_count + 1
                    if exec_state.error_count == settings.MAX_ERROR_COUNT:
                        ret_dict['msg'] = settings.UPROMPT[7]
                        ret_dict['chat_id'] = -1
                        save_on_redis = False
                    else:
                        ret_dict['msg'] = settings.UPROMPT[11]
            else:
                exec_state.error_count = exec_state.error_count + 1
                if exec_state.error_count == settings.MAX_ERROR_COUNT:
                    ret_dict['msg'] = settings.UPROMPT[7]
                    ret_dict['chat_id'] = -1
                    save_on_redis = False
                else:
                    ret_dict['msg'] = settings.UPROMPT[3]
            
        elif exec_state.state == 2:
            if exec_state.iter_deepening_search(msg):
                success, solution = exec_state.get_problem_solution()
                if success:
                    ret_dict['msg'] = 'Sugestão: ' + solution + '.\n\nResolveu o seu problema? (sim/não)'
                    exec_state.state = exec_state.state + 1
                    exec_state.error_count = 0
                else:
                    ret_dict['msg'] = solution
            else:
                exec_state.error_count = exec_state.error_count + 1
                if exec_state.error_count == settings.MAX_ERROR_COUNT:
                    ret_dict['msg'] = settings.UPROMPT[7]
                    ret_dict['chat_id'] = -2
                    save_on_redis = False
                else:
                    ret_dict['msg'] = settings.UPROMPT[6]

        elif exec_state.state == 3:
            options_array = [('sim', 1), ('s', 1), ('não', 0), ('nao', 0), ('n', 0)]
            match_option = [x[1] for x in options_array if msg.lower() == x[0]]
            if match_option:
                option = match_option[0]
                if option: # problem has been resolved
                    ret_dict['msg'] = settings.UPROMPT[9]
                    ret_dict['chat_id'] = -1
                    save_on_redis = False
                    
                    # write to log file of problems
                    save_to_log(exec_state)
                else: # problem has not been resolved
                    if len(exec_state.suggestions) > exec_state.suggestion_count + 1: # there are still more suggestions to make
                        exec_state.suggestion_count = exec_state.suggestion_count + 1
                        new_suggestion = exec_state.suggestions[exec_state.suggestion_count]['prediction']
                        ret_dict['msg'] = 'Outra sugestão: ' + new_suggestion + '.\n\nResolveu o seu problema? (sim/não)'
                    else: # no more suggestions
                        ret_dict['msg'] = settings.UPROMPT[10]
                        ret_dict['chat_id'] = -2
                        save_on_redis = False
            else:
                exec_state.error_count = exec_state.error_count + 1
                if exec_state.error_count == settings.MAX_ERROR_COUNT:
                    ret_dict['msg'] = settings.UPROMPT[7]
                    ret_dict['chat_id'] = -1
                    save_on_redis = False
                else:
                    ret_dict['msg'] = settings.UPROMPT[8]

        if save_on_redis:
            cs['content'].append(msg)        
            cs['state'] = base64.encodebytes(pickle.dumps(exec_state)).decode()
    else:
        ret_dict['msg'] = settings.UPROMPT[0]
        exec_state = ids.Inter_State()
        cs = {'content': [msg],
              'state': base64.encodebytes(pickle.dumps(exec_state)).decode()}

    if save_on_redis:
        redis_db.set(chat_id+'_ntp', json.dumps(cs).encode())
    else:
        redis_db.delete(chat_id+'_ntp')

    return app.response_class(
        response=json.dumps(ret_dict),
        mimetype='application/json'
    )


def save_to_log(exec_state):
    ''' Save problems received and suggestions made
    :param: execution state of conversation
    '''
    log = open(settings.FILENAME, "a")

    resp_array = [None] * 8
    # Serviço
    resp_array[0] = exec_state.service
    # Equipamento
    resp_array[1] = exec_state.equipment
    # Tarifário
    resp_array[2] = exec_state.tariff
    # Sintoma
    resp_array[3] = exec_state.model_args['Sintoma'][0]
    # Tipificações 1, 2 e 3
    resp_array[4] = exec_state.model_args['Tipificacao_Nivel_1'][0]
    resp_array[5] = exec_state.model_args['Tipificacao_Nivel_2'][0]
    resp_array[6] = exec_state.model_args['Tipificacao_Nivel_3'][0]
    # Sugestão
    resp_array[7] = exec_state.suggestion

    log.write(';'.join(resp_array) + '\n')
    log.close()

def loadSettings():
    ''' Loads default settings from env variables
    '''
    s_host = os.getenv('SOLVER_HOST', '127.0.0.1')
    s_port= os.getenv('SOLVER_PORT', '8000')
    settings.SOLVER_ENDPOINT = settings.SOLVER_ENDPOINT.format(s_host, s_port)
    settings.SOLVER_ENDPOINT_LOGIN = settings.SOLVER_ENDPOINT_LOGIN.format(s_host, s_port)
    settings.SOLVER_ENDPOINT_SOLVE = settings.SOLVER_ENDPOINT_SOLVE.format(s_host, s_port)
    settings.SOLVER_ENDPOINT_SERVICE_CHECK = settings.SOLVER_ENDPOINT_SERVICE_CHECK.format(s_host, s_port)
    settings.SOLVER_ENDPOINT_UPDATE_LOG = settings.SOLVER_ENDPOINT_UPDATE_LOG.format(s_host, s_port)
    settings.REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
    settings.REDIS_PORT = os.getenv('REDIS_PORT', 6379)
    
def upload_csv():
    files = {'problems_log': open(settings.FILENAME, 'rb')}
    r = requests.post(settings.SOLVER_ENDPOINT_UPDATE_LOG, files=files)
 

if __name__ == '__main__':
    # start csv file
    log = open(settings.FILENAME, "w")
    log.write('Servico;Equipamento_Tipo;Tarifario;Sintoma;Tipificacao_Nivel_1;Tipificacao_Nivel_2;Tipificacao_Nivel_3;Contexto_Saida\n')
    log.close()

    loadSettings()
    
    # redis connection
    redis_db = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)
    
    # load model for sentences similarity
    msg_interpreter.loadModelData()

    scheduler = BackgroundScheduler()
    scheduler.add_job(upload_csv, 'cron', hour=4, minute=0)
    scheduler.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)
