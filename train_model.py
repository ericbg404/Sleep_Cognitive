"""
Trabalho de RNAs - MLP para Previsao de Desempenho Cognitivo (Sono)
====================================================================

Este script e a uniao de TODAS as celulas do notebook "06 - RNA v5 FINAL.ipynb"
em um unico arquivo executavel, na mesma ordem logica do notebook original:

  1. Imports e reprodutibilidade
  2. Carregamento dos dados (Kaggle)
  3. Selecao de variaveis
  4. Divisao treino / validacao / teste (70/10/20)
  5. Pre-processamento (StandardScaler + RobustScaler)
  6. Funcao de treino reutilizavel (treinar_mlp)
  7. Bateria inicial de experimentos
  8. Testes de ablacao
  9. Rodada de combinacoes ("mais poderoso")
  10. Escolha do modelo final
  11. Avaliacao visual do modelo final
  12. Funcao de explicacao individual (prever_com_explicacao)
  13. Salvamento dos artefatos finais (usados pelo backend/app.py)

COMO RODAR
----------
Precisa de acesso a internet para baixar o dataset do Kaggle (kagglehub),
por isso o ideal e rodar no Google Colab ou em uma maquina com as
credenciais do Kaggle configuradas (~/.kaggle/kaggle.json).

    pip install -r requirements_train.txt
    python train_model.py

No final, este script gera 3 arquivos na pasta atual:
    - modelo_mlp_final.keras
    - preprocessador_final.pkl
    - dados_treino_referencia.pkl

Copie esses 3 arquivos para a pasta `backend/artifacts/` para que o
backend (app.py) consiga carregar o modelo e servir o frontend.
"""

import matplotlib

matplotlib.use("Agg")  # roda sem precisar de tela (headless)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 2. Carregamento dos dados
# ---------------------------------------------------------------------------
def carregar_dados():
    import kagglehub

    path = kagglehub.dataset_download(
        "mohankrishnathalla/sleep-health-and-daily-performance-dataset"
    )
    file_path = f"{path}/sleep_health_dataset.csv"

    df = pd.read_csv(file_path)
    df.info()
    return df


# ---------------------------------------------------------------------------
# 3. Selecao de variaveis (sem codificar ainda)
# ---------------------------------------------------------------------------
colunas_numericas = [
    "sleep_duration_hrs",
    "exercise_day",
    "sleep_latency_mins",
    "wake_episodes_per_night",
    "alcohol_units_before_bed",
    "bmi",
]
coluna_categorica = ["mental_health_condition"]
colunas_finais = colunas_numericas + coluna_categorica
target_col = "cognitive_performance_score"


# ---------------------------------------------------------------------------
# 6. Funcao de treino reutilizavel
# ---------------------------------------------------------------------------
def treinar_mlp(
    camadas,
    dropout=None,
    l2=None,
    ativacao="relu",
    lr=0.001,
    batch_size=256,
    epochs=200,
    usar_reduce_lr=False,
    nome="modelo",
    patience=10,
    verbose=0,
    dados=None,
):
    tf.random.set_seed(SEED)

    if dados is None:
        X_tr, y_tr = X_train_scaled, y_train
        X_va, y_va = X_val_scaled, y_val
        X_te, y_te = X_test_scaled, y_test
        preproc_usado = "padrao"
    else:
        X_tr, y_tr, X_va, y_va, X_te, y_te = dados
        preproc_usado = "robusto"

    regularizador = keras.regularizers.l2(l2) if l2 else None

    camadas_modelo = [layers.Input(shape=(X_tr.shape[1],))]
    for n in camadas:
        camadas_modelo.append(
            layers.Dense(n, activation=ativacao, kernel_regularizer=regularizador)
        )
        if dropout:
            camadas_modelo.append(layers.Dropout(dropout))
    camadas_modelo.append(layers.Dense(1))

    model = keras.Sequential(camadas_modelo)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]
    if usar_reduce_lr:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=verbose
            )
        )

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    y_pred = model.predict(X_te, verbose=0).flatten()
    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    gap_overfit = history.history["loss"][-1] - history.history["val_loss"][-1]

    print(
        f"[{nome}] epocas reais: {len(history.history['loss'])}/{epochs} | "
        f"R2={r2:.4f} | MAE={mae:.4f} | gap treino-val={gap_overfit:.2f}"
    )

    return {
        "nome": nome,
        "camadas": camadas,
        "dropout": dropout,
        "l2": l2,
        "ativacao": ativacao,
        "lr": lr,
        "batch_size": batch_size,
        "usar_reduce_lr": usar_reduce_lr,
        "epocas_reais": len(history.history["loss"]),
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "gap_overfit": gap_overfit,
        "history": history,
        "model": model,
        "preprocessador_usado": preproc_usado,
    }


def monta_tabela(lista_resultados):
    return (
        pd.DataFrame(
            [
                {
                    "Modelo": r["nome"],
                    "Camadas": r["camadas"],
                    "Dropout": r["dropout"],
                    "L2": r["l2"],
                    "Ativacao": r["ativacao"],
                    "LR": r["lr"],
                    "Batch": r["batch_size"],
                    "ReduceLR": r["usar_reduce_lr"],
                    "Pre-proc.": r["preprocessador_usado"],
                    "Epocas reais": r["epocas_reais"],
                    "R2": round(r["r2"], 4),
                    "MAE": round(r["mae"], 4),
                    "Gap treino-val": round(r["gap_overfit"], 2),
                }
                for r in lista_resultados
            ]
        )
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 13. Explicacao individual - "quais fatores mais pesaram para ESSA pessoa"
# ---------------------------------------------------------------------------
def prever_com_explicacao(
    modelo,
    preprocessador,
    entrada_usuario,
    dados_treino_originais,
    colunas_numericas,
    coluna_categorica,
    top_n=3,
):
    """
    entrada_usuario: dict com as 7 chaves originais, ex.:
        {'sleep_duration_hrs': 6.5, 'exercise_day': 1, 'sleep_latency_mins': 20,
         'wake_episodes_per_night': 2, 'alcohol_units_before_bed': 1.0, 'bmi': 24.3,
         'mental_health_condition': 'Healthy'}
    dados_treino_originais: X_train (DataFrame, ANTES da transformacao) - usado para calcular
        a media/moda de referencia de cada variavel
    Retorna um dict com: nota_prevista, top_fatores (lista ordenada), todos_fatores (Series completa)
    """
    colunas_todas = colunas_numericas + coluna_categorica
    entrada_df = pd.DataFrame([entrada_usuario])[colunas_todas]

    nota_prevista = modelo.predict(preprocessador.transform(entrada_df), verbose=0).flatten()[
        0
    ]

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
        "todos_fatores": serie_contribuicoes.round(2).to_dict(),
    }


# ---------------------------------------------------------------------------
# MAIN - roda o pipeline inteiro, do jeito que estava no notebook
# ---------------------------------------------------------------------------
def main():
    global X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    # 2. dados
    df = carregar_dados()

    # 3. selecao de variaveis
    X = df[colunas_finais].copy()
    y = df[target_col].values
    print(
        "Categorias existentes em mental_health_condition:",
        df["mental_health_condition"].unique(),
    )

    # 4. divisao treino / validacao / teste (70/10/20)
    y_binned = pd.qcut(y, q=5, labels=False)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y_binned
    )
    y_train_val_binned = pd.qcut(y_train_val, q=5, labels=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=SEED, stratify=y_train_val_binned
    )
    print(f"Treino: {len(X_train)}  Validacao: {len(X_val)}  Teste: {len(X_test)}")

    # 5. pre-processamento padrao (StandardScaler)
    preprocessador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colunas_numericas),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), coluna_categorica),
        ]
    )
    X_train_scaled = preprocessador.fit_transform(X_train)
    X_val_scaled = preprocessador.transform(X_val)
    X_test_scaled = preprocessador.transform(X_test)
    print("Total de features:", X_train_scaled.shape[1])

    # 5b. pre-processador alternativo (RobustScaler)
    preprocessador_robusto = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), colunas_numericas),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), coluna_categorica),
        ]
    )
    X_train_robusto = preprocessador_robusto.fit_transform(X_train)
    X_val_robusto = preprocessador_robusto.transform(X_val)
    X_test_robusto = preprocessador_robusto.transform(X_test)

    # 7. bateria inicial de experimentos
    configuracoes = [
        dict(camadas=[64, 32], dropout=0.2, lr=0.001, epochs=200, nome="base_com_dropout"),
        dict(camadas=[32, 16], dropout=None, lr=0.001, epochs=200, nome="simples_sem_dropout"),
        dict(camadas=[32, 16], dropout=0.1, lr=0.001, epochs=200, nome="simples_dropout_leve"),
        dict(camadas=[64, 32, 16], dropout=0.1, lr=0.001, epochs=200, nome="moderado_3camadas"),
        dict(
            camadas=[32, 16], dropout=None, lr=0.005, epochs=200, nome="simples_lr_intermediario"
        ),
        dict(camadas=[32, 16], dropout=None, lr=0.0005, epochs=300, nome="simples_lr_menor"),
    ]
    resultados = [treinar_mlp(**cfg, verbose=0) for cfg in configuracoes]
    tabela_inicial = monta_tabela(resultados)
    print(tabela_inicial)

    # 8. testes de ablacao (uma tecnica de cada vez, a partir do melhor da secao 7)
    melhor_inicial = tabela_inicial.iloc[0]
    config_base = next(cfg for cfg in configuracoes if cfg["nome"] == melhor_inicial["Modelo"])
    print("Configuracao base para ablacao:", config_base)

    teste1 = treinar_mlp(**{**config_base, "batch_size": 32, "nome": "teste1_batch32"}, verbose=0)
    teste2 = treinar_mlp(
        **{**config_base, "usar_reduce_lr": True, "epochs": 200, "nome": "teste2_reduce_lr"},
        verbose=0,
    )
    teste3 = treinar_mlp(**{**config_base, "ativacao": "swish", "nome": "teste3_swish"}, verbose=0)
    teste4 = treinar_mlp(
        **{**config_base, "nome": "teste4_robust_scaler"},
        dados=(X_train_robusto, y_train, X_val_robusto, y_val, X_test_robusto, y_test),
        verbose=0,
    )
    teste5 = treinar_mlp(
        **{**config_base, "dropout": None, "l2": 0.01, "nome": "teste5_l2"}, verbose=0
    )
    ablacao = [teste1, teste2, teste3, teste4, teste5]
    tabela_ablacao = monta_tabela(ablacao)
    print(tabela_ablacao)

    # 10. rodada final: combinando as melhores tecnicas
    r2_base = melhor_inicial["R2"]
    tecnicas_vencedoras = tabela_ablacao[tabela_ablacao["R2"] > r2_base]["Modelo"].tolist()
    print("Tecnicas que isoladamente melhoraram o R2 em relacao ao baseline:", tecnicas_vencedoras)

    combo_tudo = treinar_mlp(
        **{
            **config_base,
            "batch_size": 32,
            "usar_reduce_lr": True,
            "ativacao": "swish",
            "dropout": None,
            "l2": 0.01,
            "nome": "combo_tudo",
        },
        dados=(X_train_robusto, y_train, X_val_robusto, y_val, X_test_robusto, y_test),
        verbose=0,
    )
    combo_sem_l2 = treinar_mlp(
        **{
            **config_base,
            "batch_size": 32,
            "usar_reduce_lr": True,
            "ativacao": "swish",
            "nome": "combo_sem_l2",
        },
        dados=(X_train_robusto, y_train, X_val_robusto, y_val, X_test_robusto, y_test),
        verbose=0,
    )
    combo_leve = treinar_mlp(
        **{**config_base, "usar_reduce_lr": True, "ativacao": "swish", "nome": "combo_leve"},
        verbose=0,
    )
    combinacoes = [combo_tudo, combo_sem_l2, combo_leve]
    tabela_combos = monta_tabela(combinacoes)
    print(tabela_combos)

    # 11. escolha do modelo final
    todos_resultados = resultados + ablacao + combinacoes
    tabela_geral = monta_tabela(todos_resultados)
    print(tabela_geral)

    melhor = tabela_geral.iloc[0]
    print("\nModelo final escolhido:", melhor["Modelo"])

    modelo_final_dict = next(r for r in todos_resultados if r["nome"] == melhor["Modelo"])
    modelo_final = modelo_final_dict["model"]

    if modelo_final_dict["preprocessador_usado"] == "robusto":
        preprocessador_vencedor = preprocessador_robusto
        X_test_scaled_vencedor = X_test_robusto
    else:
        preprocessador_vencedor = preprocessador
        X_test_scaled_vencedor = X_test_scaled

    print("Preprocessador do modelo vencedor:", modelo_final_dict["preprocessador_usado"])

    # 12. avaliacao visual do modelo final (salva um PNG em vez de mostrar na tela)
    y_pred_final = modelo_final.predict(X_test_scaled_vencedor, verbose=0).flatten()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    h = modelo_final_dict["history"].history
    axes[0].plot(h["loss"], label="Treino")
    axes[0].plot(h["val_loss"], label="Validacao")
    axes[0].set_title("Curva de aprendizado (MSE)")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    axes[1].scatter(y_test, y_pred_final, alpha=0.2, s=8)
    lims = [min(y_test.min(), y_pred_final.min()), max(y_test.max(), y_pred_final.max())]
    axes[1].plot(lims, lims, "r--", label="Previsao perfeita")
    axes[1].set_xlabel("Valor real")
    axes[1].set_ylabel("Valor previsto")
    axes[1].set_title(f'Previsto x Real (R2={melhor["R2"]:.4f})')
    axes[1].legend()

    residuos = y_test - y_pred_final
    axes[2].scatter(y_pred_final, residuos, alpha=0.2, s=8)
    axes[2].axhline(0, color="r", linestyle="--")
    axes[2].set_xlabel("Valor previsto")
    axes[2].set_ylabel("Residuo (real - previsto)")
    axes[2].set_title("Residuos")
    plt.tight_layout()
    plt.savefig("avaliacao_modelo_final.png", dpi=150)
    plt.close(fig)

    # 13. demonstracao rapida da funcao de explicacao individual
    exemplo_usuario = {
        "sleep_duration_hrs": 5.5,
        "exercise_day": 0,
        "sleep_latency_mins": 35,
        "wake_episodes_per_night": 3,
        "alcohol_units_before_bed": 2.0,
        "bmi": 27.0,
        "mental_health_condition": "Anxiety",
    }
    resultado_exemplo = prever_com_explicacao(
        modelo_final,
        preprocessador_vencedor,
        exemplo_usuario,
        X_train,
        colunas_numericas,
        coluna_categorica,
        top_n=3,
    )
    print("Nota prevista (exemplo):", resultado_exemplo["nota_prevista"])
    for f in resultado_exemplo["top_fatores"]:
        print(f"  - {f['variavel']}: {f['contribuicao']:+.2f} ({f['direcao']})")

    # 14. salvando os artefatos finais (modelo + preprocessador + dados de referencia)
    modelo_final.save("modelo_mlp_final.keras")
    joblib.dump(preprocessador_vencedor, "preprocessador_final.pkl")
    joblib.dump(X_train, "dados_treino_referencia.pkl")

    print("\nModelo, preprocessador e dados de referencia salvos na pasta atual.")
    print("Copie os 3 arquivos .keras/.pkl para backend/artifacts/ e rode o backend (app.py).")


if __name__ == "__main__":
    main()
