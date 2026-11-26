# Análise Experimental por Troca de Módulos no iTransformer (ICLR 2024)

Trabalho Final da disciplina **Aprendizado de Máquina Profundo** (UFSM — Prof. Thiago L. T. da Silveira).  
**Grupo:** André Renner, Andrei Hammacher, Rafael Wagner.

## Artigo e Código Base
- **Artigo:** *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting* (Liu et al., ICLR 2024) — [arXiv:2310.06625](https://arxiv.org/abs/2310.06625)
- **Repositório oficial fixado:** `thuml/iTransformer` @ commit `c2426e68ca13f74aaec08045c5c724d8ad328124`

## Estrutura do Repositório
- `iTransformer_ablation.ipynb` — Notebook autocontido (Kaggle / Colab Pro) que clona o repositório oficial no commit fixado, aplica as modificações (`# [MOD]`), executa o teste de equivalência do baseline, baixa os dados (`ETTh1` e `Weather`), roda os 54 treinos (9 variantes × 2 bases × 3 seeds) com retomada automática via JSONL e gera tabelas e gráficos.
- `modifications.patch` — `git diff` completo das alterações aplicadas ao código oficial (todas marcadas com `# [MOD]`).
- `relatorio/main.tex` e `relatorio/tabela_resultados.tex` — Relatório em formato `IEEEtran` (`conference`, duas colunas, ≤ 2 páginas sem referências), pronto para compilação no Overleaf.
- `slides_roteiro.md` — Roteiro resumido em Markdown alinhado à estrutura do relatório.
- `apresentacao_iTransformer.pptx` — Apresentação de 15 slides em formato 16:9 (dividida igualmente: 5 slides por integrante) com gráficos 300 dpi gerados a partir dos 54 treinos.
- `roteiro_pitch_iTransformer.docx` — Roteiro completo do pitch em Word com foto de cada um dos 15 slides, frase curta, fala palavra por palavra, explicação completa e técnica, dicas de apresentação e Q&A para a banca.
- `resultados/` — Resultados finais dos 54 treinos em GPU Tesla T4 (`results_full.jsonl`, `summary_results.csv`, `ablation_ETTh1.png` e `ablation_Weather.png`).
