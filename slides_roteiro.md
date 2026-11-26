# Roteiro de Apresentação — Trabalho Final de Aprendizado de Máquina Profundo (UFSM)

**Artigo:** *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting* (Liu et al., ICLR 2024)  
**Duração-alvo:** 10–12 minutos (11 slides, ~1 min/slide)  
**Estrutura:** Espelha integralmente o relatório `relatorio/main.tex`

---

## Slide 1 — Capa (0:30)
- **Título:** Análise Experimental por Troca de Módulos no iTransformer
- **Subtítulo:** Onde e como operam os módulos de Deep Learning na previsão de séries temporais multivariadas
- **Metadados:** Nome do aluno | UFSM — Aprendizado de Máquina Profundo | Prof. Thiago L. T. da Silveira | Dezembro/2026
- **Fala-chave:** Apresentar rapidamente o objetivo do trabalho: não é bater recorde de MSE, e sim abrir a caixa-preta do iTransformer trocando sistematicamente cada módulo avançado para verificar seu papel real.

---

## Slide 2 — Motivação e Ideia Central do iTransformer (1:00)
- **Problema do Transformer padrão em séries temporais:**
  - Token = instante temporal $t$ (agrupa todas as $N$ variáveis num único vetor).
  - Mistura variáveis com escalas/unidades diferentes e limita o campo receptivo temporal.
- **Solução do iTransformer (ICLR 2024):**
  - Mantém o codificador Transformer intacto, mas **inverte** os eixos: cada **variável inteira** ($L$ passos) vira um token ($B \times L \times N \to B \times N_{\text{tok}} \times E$).
  - Atenção aprende **correlações entre variáveis**; FFN aprende **representação temporal** de cada série.
- **Visual sugerido:** Comparação lado a lado: *Token temporal* vs. *Token de variável (invertido)*.

---

## Slide 3 — Mapeamento dos Módulos e Fluxo de Tensores (1:15)
- **Visual principal:** Diagrama do relatório ($B \times L \times N \to B \times N_{\text{tok}} \times E \to B \times N \times S \to B \times S \times N$).
- **Fórmulas-chave na tela (enxutas):**
  1. **Normalização de instância (RevIN):** $\tilde{X}_{:,n} = (X_{:,n}-\mu_n)/\sigma_n$ + desnormalização na saída.
  2. **Embedding invertido:** $h_n^{(0)} = W_e \tilde{X}_{:,n} + b_e$ (inclui 4 covariáveis temporais como tokens extras: $N_{\text{tok}} = N + 4$).
  3. **Self-Attention entre variáveis:** $\text{softmax}(QK^\top/\sqrt{d_k})V$ com mapa $N_{\text{tok}} \times N_{\text{tok}}$.
  4. **LayerNorm por token** + **FFN ($1\times 1$ Conv + GELU)** + **Conexões residuais** + **Projeção linear** $\hat{y}_n = W_p h_n$.
- **Fala-chave:** Destacar *qual representação* cada módulo enxerga (série inteira vs. vetor de features vs. lote).

---

## Slide 4 — Desenho Experimental e Controle de Qualidade (1:00)
- **Reprodutibilidade:** código oficial `thuml/iTransformer` fixado no commit `c2426e6`, sem alterar a ordem de inicialização dos parâmetros no `AblationEncoderLayer`.
  - **Teste de equivalência:** saída do modelo modificado em modo `baseline` é numericamente idêntica à do original com a mesma seed.
- **Recorte experimental (justificado por tempo/GPU):**
  - Lookback $L = 96$, horizonte único $S = 96$.
  - 2 bases: **ETTh1** ($N=7$, $N_{\text{tok}}=11$, $E=256$, 2 camadas) e **Weather** ($N=21$, $N_{\text{tok}}=25$, $E=512$, 3 camadas).
  - **3 seeds** (2021, 2022, 2023) $\times$ **9 variantes** $\times$ **2 bases** = **54 treinos**.
- **Métricas:** MSE, MAE (média $\pm$ desvio), $\Delta\%$ vs. baseline, nº de parâmetros, tempo/época e épocas até *early stopping*.

---

## Slide 5 — As 9 Configurações (Mapa das Trocas de Módulos) (1:00)
- **Tabela visual resumida por família:**
  - **Baseline:** iTransformer original.
  - **Atenção entre variáveis:** `attn_none` (remove atenção) e `attn_linear` ($W_o \cdot M \cdot (X W_v)$, mistura estática independente da entrada).
  - **Normalização da representação:** `norm_batch` (BatchNorm1d em $B \cdot N_{\text{tok}}$ tokens) e `norm_none` (Identity) — em `norm1`, `norm2` e `norm_out`.
  - **Normalização de instância:** `revin_none` (desliga) e `revin_center` (só remove/soma média, sem desvio).
  - **FFN:** `ffn_linear` (substitui MLP+GELU por `nn.Linear(E, E)`).
  - **Resíduos:** `no_residual` (remove ambos os atalhos residuais do encoder layer).
- **Fala-chave:** Explicar rapidamente o papel da *hipótese* que cada troca testa.

---

## Slide 6 — Reprodução do Baseline vs. Artigo Original (0:45)
- **Tabela curta de conferência ($L=96, S=96$):**
  - **ETTh1:** Artigo MSE `0.386` / MAE `0.405` *(conferir no PDF)* vs. Nosso Baseline `[PREENCHER]`.
  - **Weather:** Artigo MSE `0.174` / MAE `0.214` *(conferir no PDF)* vs. Nosso Baseline `[PREENCHER]`.
- **Fala-chave:** Mostrar que o baseline reproduz de forma consistente a ordem de grandeza do artigo e explicar pequenas variações (média de 3 seeds vs. seed única 2023, versão de PyTorch/CUDA).

---

## Slide 7 — Visão Geral dos Resultados (Gráficos de Barras) (1:00)
- **Visual principal:** Inserir os 2 gráficos PNG (300 dpi) gerados pelo notebook (`ablation_ETTh1.png` e `ablation_Weather.png`), mostrando MSE e MAE com barras de erro ($\pm$ desvio) e linha tracejada no baseline.
- **Destaque visual:** Marcar quais diferenças são menores que o desvio-padrão do baseline (dentro da faixa de ruído).
- **Fala-chave:** Dar o panorama antes de entrar família por família: "quais módulos quebram o modelo se removidos e quais quase não mudam o erro?"

---

## Slide 8 — Análise: Atenção entre Variáveis (`attn_none` vs. `attn_linear`) (1:15)
- **Pergunta científica:** Misturar variáveis importa? E precisa ser *dependente da entrada* (softmax) ou uma matriz estática $M \in \mathbb{R}^{N_{\text{tok}} \times N_{\text{tok}}}$ basta?
- **Hipótese verificada nos dados:**
  - Comparar $\Delta\%$ em **Weather** ($N=21$) vs. **ETTh1** ($N=7$): `[PREENCHER com valores reais]`.
  - Comparar `attn_linear` vs. `baseline` e `attn_none`: `[PREENCHER: se attn_linear aproximou o baseline ou ficou atrás]`.
- **Custo:** Comparação de parâmetros e tempo/época (sem $W_q, W_k$ no `attn_linear`, sem bloco inteiro no `attn_none`).

---

## Slide 9 — Análise: Normalizações (Instância vs. Representação) (1:15)
- **Normalização de instância (`revin_none`, `revin_center`):**
  - Impacto da não estacionariedade (mudança de nível e variância entre janelas): `[PREENCHER valores de ETTh1 e Weather]`.
  - Separar efeito de centralizar (`revin_center`) vs. escalar pelo desvio (`baseline`).
- **Normalização da representação (`norm_batch`, `norm_none`):**
  - *LayerNorm* (por série) vs. *BatchNorm* (por lote de $B \cdot N_{\text{tok}}$ tokens): por que estatísticas de lote podem degradar (ou não) séries multivariadas heterogêneas.
  - `[PREENCHER valores e estabilidade de convergência]`.

---

## Slide 10 — Análise: FFN, Conexões Residuais e Custo Computacional (1:00)
- **FFN (`ffn_linear`):**
  - Remover não linearidade (GELU + expansão $d_{\text{ff}}$): `[PREENCHER impacto no MSE/MAE e redução de parâmetros]`.
- **Conexões residuais (`no_residual`):**
  - Impacto em 2 camadas (ETTh1) vs. 3 camadas (Weather): `[PREENCHER erro e nº de épocas até early stopping]`.
- **Resumo de custo:** Tabela compacta comparando parâmetros treináveis e segundos/época nas duas bases (`[PREENCHER]`).

---

## Slide 11 — Limitações, Conclusões e Lições Aprendidas (1:00)
- **Limitações honestas:**
  - Horizonte único ($S=96$), 2 bases (baixa/média dimensionalidade), 3 seeds — efeitos de atenção podem ser mais pronunciados em bases grandes (ex.: Traffic, $N=862$) ou horizontes longos ($S=720$).
- **Conclusões principais:**
  - Ranking dos módulos por impacto real no MSE/MAE: `[PREENCHER ordem observada nos dados]`.
  - Conexão com a disciplina: a inversão de tokens do iTransformer reposiciona *o que* cada módulo de Deep Learning faz — e a ablação controlada permite separar intuição arquitetural de evidência empírica.
- **Encerramento:** Agradecimento e abertura para perguntas.
