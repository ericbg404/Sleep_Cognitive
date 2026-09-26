# NoitBem — Previsão de Desempenho Cognitivo a partir do Sono

Projeto reorganizado a partir do notebook `06_-_RNA_v5_FINAL.ipynb`. Estrutura:

```
sleep_cognitive_app/
├── train_model.py            # 1) TODAS as células do notebook, unificadas
├── requirements_train.txt    # dependências para treinar
├── backend/
│   ├── app.py                # 3) API Flask (carrega o modelo e prevê)
│   ├── model_utils.py        # função prever_com_explicacao (Seção 13 do notebook)
│   ├── requirements.txt      # dependências do servidor
│   └── artifacts/            # <- coloque aqui os 3 arquivos gerados pelo treino
└── frontend/
    └── index.html            # 2) e 4) formulário + resultado, visual moderno
```

## Passo 1 — Gerar o modelo (`train_model.py`)

Esse é o notebook inteiro (Seções 1 a 14) em um único script. Ele baixa o dataset do
Kaggle via `kagglehub`, roda a bateria de experimentos, a ablação, as combinações,
escolhe o melhor modelo e salva os 3 artefatos.

Como precisa de acesso ao Kaggle, rode no **Google Colab** (mesmo ambiente do notebook
original) ou localmente com suas credenciais do Kaggle configuradas
(`~/.kaggle/kaggle.json`):

```bash
pip install -r requirements_train.txt
python train_model.py
```

Isso gera, na pasta onde rodou:
- `modelo_mlp_final.keras`
- `preprocessador_final.pkl`
- `dados_treino_referencia.pkl`

**Copie esses 3 arquivos para `backend/artifacts/`** (crie a pasta se não existir).

## Passo 2 — Rodar o backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

O servidor sobe em `http://localhost:5000` e já serve o frontend automaticamente
(não precisa abrir o `index.html` direto — abra pelo navegador em `http://localhost:5000`).

- `GET /api/meta` — devolve as faixas reais (min/máx/média) de cada campo numérico e as
  categorias válidas de `mental_health_condition`, direto dos dados de treino. O
  frontend usa isso para configurar os controles do formulário automaticamente —
  então não é preciso adivinhar quais categorias existem no dataset.
- `POST /api/predict` — recebe os 7 campos em JSON e devolve `nota_prevista` (0–100) e
  os fatores que mais pesaram para aquela pessoa (mesma lógica da Seção 13 do notebook).

## Passo 3 — Usar

Abra `http://localhost:5000`, preencha:

- Duração do sono (h)
- Se se exercitou naquele dia
- Tempo para pegar no sono (min)
- Despertares durante a noite
- Álcool antes de dormir (doses)
- IMC
- Condição de saúde mental

Clique em **"Calcular minha nota"** — o velocímetro mostra a nota prevista (0–100) e a
lista abaixo mostra, em ordem, quais fatores mais empurraram sua nota para cima (verde)
ou para baixo (coral), em relação à média dos dados de treino.

## Observações

- O `train_model.py` só precisa ser rodado **uma vez** (ou sempre que quiser retreinar).
  O dia a dia do app é só backend + frontend, usando os artefatos já salvos.
- Se abrir o frontend sem o backend rodando (ou sem os artefatos na pasta
  `backend/artifacts/`), o formulário ainda aparece, mas a previsão retorna um erro
  explicando o que falta.
- Todo o pré-processamento (`StandardScaler`/`RobustScaler` + `OneHotEncoder`, conforme
  o que venceu no treino) é aplicado automaticamente pelo backend — o usuário final só
  vê os 7 campos originais, nunca as colunas transformadas.
