import json, redis
import settings
import msg_interpreter
import ids
from flask import Flask, request

app = Flask(__name__)

@app.route('/solver', methods=['POST'])
def solve():
    data = request.get_json()
    idChat = data['idChat']
    msg = data['msg']
    rt = {'idChat': idChat, 'msg':''}
    csr = redis_db.get(idChat)
    if csr:
        cs = json.loads(csr.decode('utf-8'))
        cs['content'].append(msg)
        rt['msg'] = settings.UPROMPT[1]
        ids.iter_deepening_search(msg, 'Internet')
    else:
        cs = {'content': [msg]}
        rt['msg'] = settings.UPROMPT[0]
    rts = json.dumps(rt)
    redis_db.set(idChat, json.dumps(cs).encode())
    return app.response_class(
        response=rts,
        mimetype='application/json'
    )

if __name__ == '__main__':
    redis_db = redis.Redis(host='127.0.0.1', port=6379, db=0)
    msg_interpreter.loadModelData()
    app.run(port=5000, threaded=True)
