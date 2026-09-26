"""
Funcoes e constantes compartilhadas com o notebook de treino.
Mantidas identicas a Secao 13 do notebook (06 - RNA v5 FINAL) para garantir
que o resultado da API seja exatamente o mesmo que o do notebook.
"""

import pandas as pd

# Mesma ordem/nomes de colunas usados no treino (Secao 3 do notebook)
COLUNAS_NUMERICAS = [
    "sleep_duration_hrs",
    "exercise_day",
    "sleep_latency_mins",
    "wake_episodes_per_night",
    "alcohol_units_before_bed",
    "bmi",
]
COLUNA_CATEGORICA = ["mental_health_condition"]
COLUNAS_TODAS = COLUNAS_NUMERICAS + COLUNA_CATEGORICA


def prever_com_explicacao(
    modelo,
    preprocessador,
    entrada_usuario,
    dados_treino_originais,
    colunas_numericas=COLUNAS_NUMERICAS,
    coluna_categorica=COLUNA_CATEGORICA,
    top_n=3,
):
    """
    entrada_usuario: dict com as 7 chaves originais, ex.:
        {'sleep_duration_hrs': 6.5, 'exercise_day': 1, 'sleep_latency_mins': 20,
         'wake_episodes_per_night': 2, 'alcohol_units_before_bed': 1.0, 'bmi': 24.3,
         'mental_health_condition': 'Healthy'}
    dados_treino_originais: X_train (DataFrame, ANTES da transformacao) - usado para
        calcular a media/moda de referencia de cada variavel.

    Retorna um dict com: nota_prevista, top_fatores (lista ordenada), todos_fatores.
    """
    colunas_todas = colunas_numericas + coluna_categorica
    entrada_df = pd.DataFrame([entrada_usuario])[colunas_todas]

    nota_prevista = modelo.predict(preprocessador.transform(entrada_df), verbose=0).flatten()[0]

    contribuicoes = {}
    for col in colunas_todas:
        entrada_modificada = entrada_df.copy()
        if col in colunas_numericas:
            entrada_modificada[col] = dados_treino_originais[col].mean()
        else:
            entrada_modificada[col] = dados_treino_originais[col].mode()[0]

        nota_modificada = modelo.predict(
            preprocessador.transform(entrada_modificada), verbose=0
        ).flatten()[0]

        contribuicoes[col] = float(nota_prevista - nota_modificada)

    serie_contribuicoes = pd.Series(contribuicoes).sort_values(key=abs, ascending=False)

    return {
        "nota_prevista": round(float(nota_prevista), 2),
        "top_fatores": [
            {
                "variavel": var,
                "contribuicao": round(val, 2),
                "direcao": "aumenta a nota" if val > 0 else "reduz a nota",
            }
            for var, val in serie_contribuicoes.head(top_n).items()
        ],
        "todos_fatores": {k: round(v, 2) for k, v in serie_contribuicoes.to_dict().items()},
    }
