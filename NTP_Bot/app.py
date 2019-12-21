import json
import redis
import ids
from flask import Flask, request
app = Flask(__name__)
redis_db = redis.Redis(host='127.0.0.1', port=6379, db=0)

@app.route('/solver', methods=['POST'])
def solve():
    data = request.get_json()
    idChat = data['idChat']
    msg = data['msg']
    csr = redis_db.get(idChat)
    if csr:
        cs = json.loads(csr.decode('utf-8'))
        cs['content'].append(msg)
        ids.iter_deepening_search('Internet', msg)
    else:
        cs = {'content': [msg]}
        #TODO: ask for authentication data
    response_as_json = json.dumps(cs)
    redis_db.set(idChat, json.dumps(cs).encode())
    return app.response_class(
        response=response_as_json,
        mimetype='application/json'
    )

if __name__ == '__main__':
    app.run(port=5000, threaded=True)
