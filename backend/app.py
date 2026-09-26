"""
Backend da aplicacao "Previsao de Desempenho Cognitivo (Sono)".

Carrega os 3 artefatos gerados pelo train_model.py:
    - artifacts/modelo_mlp_final.keras
    - artifacts/preprocessador_final.pkl
    - artifacts/dados_treino_referencia.pkl

E expoe:
    GET  /                -> serve o frontend (frontend/index.html)
    GET  /api/meta        -> faixas numericas e categorias validas, para montar o formulario
    POST /api/predict     -> roda prever_com_explicacao() e devolve o resultado (Secao 13)

Como rodar:
    pip install -r requirements.txt
    python app.py
    # abra http://localhost:5000
"""

import os

from flask import Flask, jsonify, request, send_from_directory
import joblib
from tensorflow import keras

from model_utils import COLUNAS_NUMERICAS, COLUNA_CATEGORICA, prever_com_explicacao

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------
# Carregamento dos artefatos (uma unica vez, na subida do servidor)
# ---------------------------------------------------------------------------
_modelo = None
_preprocessador = None
_dados_referencia = None
_erro_carregamento = None

try:
    _modelo = keras.models.load_model(os.path.join(ARTIFACTS_DIR, "modelo_mlp_final.keras"))
    _preprocessador = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessador_final.pkl"))
    _dados_referencia = joblib.load(os.path.join(ARTIFACTS_DIR, "dados_treino_referencia.pkl"))
except Exception as exc:  # noqa: BLE001 - queremos reportar qualquer erro de carregamento
    _erro_carregamento = str(exc)


# Rotulos amigaveis para exibir no frontend (nomes tecnicos -> texto em PT-BR)
ROTULOS = {
    "sleep_duration_hrs": "Duracao do sono",
    "exercise_day": "Exercicio no dia anterior",
    "sleep_latency_mins": "Tempo para pegar no sono",
    "wake_episodes_per_night": "Despertares durante a noite",
    "alcohol_units_before_bed": "Alcool antes de dormir",
    "bmi": "IMC",
    "mental_health_condition": "Condicao de saude mental",
}


def _checar_modelo_carregado():
    if _erro_carregamento or _modelo is None:
        return (
            jsonify(
                {
                    "erro": (
                        "Artefatos do modelo nao encontrados/carregados. Rode train_model.py "
                        "e copie modelo_mlp_final.keras, preprocessador_final.pkl e "
                        "dados_treino_referencia.pkl para backend/artifacts/."
                    ),
                    "detalhe": _erro_carregamento,
                }
            ),
            503,
        )
    return None


# ---------------------------------------------------------------------------
# Frontend estatico
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_assets(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.route("/api/meta")
def meta():
    """Faixas numericas (min/media/max) e categorias validas, calculadas a partir
    dos dados de treino de referencia - usado para configurar o formulario no
    frontend (sliders com faixas reais e o dropdown de condicao de saude mental)."""
    erro = _checar_modelo_carregado()
    if erro:
        return erro

    faixas = {}
    for col in COLUNAS_NUMERICAS:
        serie = _dados_referencia[col]
        faixas[col] = {
            "min": float(serie.min()),
            "max": float(serie.max()),
            "media": float(serie.mean()),
            "rotulo": ROTULOS.get(col, col),
        }

    categoria_col = COLUNA_CATEGORICA[0]
    categorias = sorted(_dados_referencia[categoria_col].dropna().unique().tolist())

    return jsonify(
        {
            "faixas_numericas": faixas,
            "categorias_saude_mental": categorias,
            "rotulos": ROTULOS,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    erro = _checar_modelo_carregado()
    if erro:
        return erro

    payload = request.get_json(silent=True) or {}

    campos_obrigatorios = COLUNAS_NUMERICAS + COLUNA_CATEGORICA
    faltando = [c for c in campos_obrigatorios if c not in payload or payload[c] in (None, "")]
    if faltando:
        return jsonify({"erro": f"Campos faltando: {', '.join(faltando)}"}), 400

    try:
        entrada_usuario = {
            "sleep_duration_hrs": float(payload["sleep_duration_hrs"]),
            "exercise_day": float(payload["exercise_day"]),
            "sleep_latency_mins": float(payload["sleep_latency_mins"]),
            "wake_episodes_per_night": float(payload["wake_episodes_per_night"]),
            "alcohol_units_before_bed": float(payload["alcohol_units_before_bed"]),
            "bmi": float(payload["bmi"]),
            "mental_health_condition": str(payload["mental_health_condition"]),
        }
    except (TypeError, ValueError):
        return jsonify({"erro": "Um ou mais valores numericos sao invalidos."}), 400

    resultado = prever_com_explicacao(
        _modelo,
        _preprocessador,
        entrada_usuario,
        _dados_referencia,
        COLUNAS_NUMERICAS,
        COLUNA_CATEGORICA,
        top_n=len(campos_obrigatorios),
    )

    # adiciona rotulos amigaveis para o frontend nao precisar traduzir
    for fator in resultado["top_fatores"]:
        fator["rotulo"] = ROTULOS.get(fator["variavel"], fator["variavel"])

    resultado["todos_fatores_rotulados"] = [
        {"variavel": var, "rotulo": ROTULOS.get(var, var), "contribuicao": val}
        for var, val in resultado["todos_fatores"].items()
    ]

    return jsonify(resultado)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
