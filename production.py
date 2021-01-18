from urllib import parse
import flask
from flask import request
import torch.utils.data
import pickle

import modules

webserver = flask.Flask("KSPZzal")
webserver.config['JSON_AS_ASCII'] = False

with open('data/model/pytorch_model.bin', 'rb') as f:
    model = pickle.load(f)


@webserver.route("/")
def flask_eval():
    resp = {}
    try:
        t = parse.unquote(request.args['t'], encoding='UTF-8')
    except KeyError:
        resp['code'] = 400
        return flask.jsonify(resp)

    eval_iterator = torch.utils.data.DataLoader(modules.ImgWithoutTemplateIterableDataset(["", ], 1),
                                                batch_size=modules.BATCH_SIZE)  # 사진 폴더

    resp['code'] = 200
    resp['data'] = {}

    with torch.no_grad():
        for batch in eval_iterator:
            resp['data']['rate'] = 1.0 - float(torch.sigmoid(model(batch.text).squeeze(1)[0]))
            return flask.jsonify(resp)


webserver.run(host="0.0.0.0", port=modules.config_get("port"))
