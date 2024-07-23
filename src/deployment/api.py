import os

import flask
import joblib
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from skfda.preprocessing.dim_reduction import FPCA

from configs.NN_config import hparams
from configs.data import DATA_FOLDER
from src.models.functional_neural_networks.MLP import MLP
from src.utils import get_project_root

root = get_project_root()
app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('index.html')


def load_model(profile):
    model = MLP()
    model.load_state_dict(
        torch.load(os.path.join(root, "logs", profile, hparams.EVAL.NAME))
    )
    model.eval()

    return model


def load_fpca(profile):
    fpca = FPCA(n_components=hparams.FPCA.N_COMPONENTS)
    fpca_dictionary = np.load(
        os.path.join(root, DATA_FOLDER, profile, f"{profile}_decomposition.npy"),
        allow_pickle="TRUE",
    ).item()

    fpca.components_ = fpca_dictionary["components_"]
    fpca.mean_ = fpca_dictionary["mean_"]
    fpca.singular_values_ = fpca_dictionary["singular_values"]

    return fpca


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    values = data['values']
    # Convert the list of values to a numpy array
    input_array = np.array(values).reshape(1, -1)
    # Scale the input array
    input_array = scaler.transform(input_array)

    prediction = []
    for model, fpca in zip([model_I, model_Q, model_U, model_V], [fpca_I, fpca_Q, fpca_U, fpca_V]):
        input_array = torch.from_numpy(input_array).to(torch.float32)
        with torch.no_grad():
            scores = model(input_array)
        scores = scores.numpy()
        I_preds = fpca.inverse_transform(scores)
        prediction.append(I_preds)

    # prediction = np.array(prediction).reshape(-1)

    return jsonify(prediction)


if __name__ == '__main__':
    model_I = load_model("I")
    model_Q = load_model("Q")
    model_U = load_model("U")
    model_V = load_model("V")

    scaler = joblib.load(os.path.join(root, DATA_FOLDER, "scaler.pkl"))

    fpca_I = load_fpca("I")
    fpca_Q = load_fpca("Q")
    fpca_U = load_fpca("U")
    fpca_V = load_fpca("V")

    app.run(host='0.0.0.0', port=8000, debug=True)
