import base64, json, pickle, redis
import settings
import msg_interpreter
import ids
import requests
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.schedulers.blocking import BlockingScheduler

app = Flask(__name__)

@app.route('/solver', methods=['POST'])
def solve():
    data = request.get_json()
    chat_id = data['idChat']
    msg = data['msg']
    
    ret_dict = {'chat_id': chat_id, 'msg':''}
    csr = redis_db.get(chat_id)

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
                elif login == 1:
                    ret_dict['msg'] = settings.UPROMPT[4]
                else:
                    ret_dict['msg'] = settings.UPROMPT[5]
            else:
                ret_dict['msg'] = settings.UPROMPT[5]
                
        elif exec_state.state == 1:
            services_array = [('tv', 'TV'), ('televisão', 'TV'), ('internet', 'Internet'), ('wifi', 'Internet'), ('voz', 'Voz')]
            match_service = [x[1] for x in services_array if msg.lower() == x[0]]
            if match_service:
                exec_state.service = match_service[0]
                ret_dict['msg'] = settings.UPROMPT[2]
                exec_state.state = exec_state.state + 1
            else:
                ret_dict['msg'] = settings.UPROMPT[3]
            
        elif exec_state.state == 2:
            if exec_state.iter_deepening_search(msg):
                success, solution = exec_state.get_problem_solution()
                if success:
                    ret_dict['msg'] = 'Sugestão: ' + solution + '.'
                else:
                    ret_dict['msg'] = solution
            else:
                ret_dict['msg'] = settings.UPROMPT[6]
        cs['content'].append(msg)        
        cs['state'] = base64.encodebytes(pickle.dumps(exec_state)).decode()        
    else:
        ret_dict['msg'] = settings.UPROMPT[0]
        exec_state = ids.Inter_State()
        cs = {'content': [msg],
              'state': base64.encodebytes(pickle.dumps(exec_state)).decode()}

    redis_db.set(chat_id, json.dumps(cs).encode())

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


def upload_csv():
    files = {'problems_log': open(settings.FILENAME, 'rb')}
    r = requests.post(settings.SOLVER_ENDPOINT_UPDATE_LOG, files)
    print(r)
    

if __name__ == '__main__':
    # start csv file
    log = open(settings.FILENAME, "w")
    log.write('Servico;Equipamento_Tipo;Tarifario;Sintoma;Tipificacao_Nivel_1;Tipificacao_Nivel_2;Tipificacao_Nivel_3;Contexto_Saida\n')
    log.close()

    # redis connection
    redis_db = redis.Redis(host='127.0.0.1', port=6379, db=0)
    
    # load model for sentences similarity
    msg_interpreter.loadModelData()

    scheduler = BackgroundScheduler()
    scheduler.add_job(upload_csv, IntervalTrigger(minutes = 3))
    scheduler.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)
