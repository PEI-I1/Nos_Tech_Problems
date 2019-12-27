import base64, json, pickle, redis
import settings
import msg_interpreter
import ids
from flask import Flask, request

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
            model_input_args = ids.iter_deepening_search(msg, exec_state.service)
            success, solution = exec_state.get_problem_solution(model_input_args)
            if success:
                ret_dict['msg'] = 'Sugestão: ' + solution + '.'
            else:
                ret_dict['msg'] = solution
            
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


if __name__ == '__main__':
    redis_db = redis.Redis(host='127.0.0.1', port=6379, db=0)
    msg_interpreter.loadModelData()
    app.run(port=5000, threaded=True)
