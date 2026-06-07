"""
Flask backend for Triagegeist.

Run from the triagegeist root:
    python webpage/app.py

Then open http://localhost:5000
"""
import sys, os
_WEBPAGE = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_WEBPAGE)
sys.path.insert(0, _WEBPAGE)

import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from predict_patient import transform_patient

MODELS_DIR = os.path.join(_WEBPAGE, "models")

print("Loading models…", flush=True)
fit_params = joblib.load(os.path.join(MODELS_DIR, "fit_params.pkl"))
lgbm_model = joblib.load(os.path.join(MODELS_DIR, "lgbm_model.pkl"))
xgb_model  = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
ALPHA = 0.25   # 25% LGBM + 75% XGB (best ensemble from tuning)
print("Ready.", flush=True)

app = Flask(__name__, static_folder=_WEBPAGE, static_url_path="")

ACUITY_LABELS = {
    1: "Resuscitation",
    2: "Emergent",
    3: "Urgent",
    4: "Less Urgent",
    5: "Non-Urgent",
}

@app.route("/")
def index():
    return send_from_directory(_WEBPAGE, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json(force=True)
    try:
        X = transform_patient(patient, fit_params)
        lgbm_proba = lgbm_model.predict_proba(X)[0]
        xgb_proba  = xgb_model.predict_proba(X)[0]
        ens_proba  = ALPHA * lgbm_proba + (1 - ALPHA) * xgb_proba
        pred       = int(ens_proba.argmax()) + 1   # 1-indexed

        return jsonify({
            "prediction":    pred,
            "label":         ACUITY_LABELS[pred],
            "probabilities": {f"L{i+1}": round(float(p), 4) for i, p in enumerate(ens_proba)},
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=False, port=5000)
