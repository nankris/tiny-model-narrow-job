# #024 Tiny local model vs Grok: news-topic classification

**Question.** For a simple, high-volume task (sorting news headlines into 4 topics), how close do tiny local models running on a CPU get to Grok, and what does each cost?

**Answer, short.** For plain classification you don't need a frontier model. With a little labelled data, a 360M model used as a frozen encoder (89.0%) or a 20-year-old TF-IDF + logistic regression (87.5%) land within 1.5–3 points of Grok (90.5%) at $0 marginal cost. Prompting the same 360M model with 4 examples does **not** work (29%: it labelled almost everything Sci/Tech). If you use Grok, put 20 items in one call: accuracy is about the same (89.0% vs 90.5%) and it costs **8.3× less** ($0.39 vs $3.20 per 1,000 items, API-equivalent).

## Setup
- **Data:** 200 items from the public AG News test split (`fancyzhx/ag_news`), 50 per class (World, Sports, Business, Sci/Tech), seed 0. 4,000 random train rows for the trained baselines. `scripts/prep.py` → `results/test200.json`, `results/train4000.json`.
- **Grok:** Grok Build CLI 1.0.52 on SuperGrok, model reported as `grok-4.7-build`. Lean flags: `--system-prompt-override "You are a news topic classifier." --disable-web-search --no-subagents --tools read_file`, 6 calls in parallel, run 2026-10-14 19:51–19:56 IST.
  - `MODE=single`: 1 item per call, 200 calls.
  - `MODE=batch`: 20 numbered items per call, answers as `<n>: <label>`, 10 calls.
- **Local (CPU only, 2 threads, 2-vCPU arm64 sandbox, 7.9 GB RAM):**
  - SmolLM2-360M-Instruct, 4-shot chat prompt (one example per class), label chosen by comparing next-token logits for the 4 label words (one forward pass per item).
  - SmolLM2-360M-Instruct as a frozen encoder: mean-pooled last hidden state → logistic regression trained on 1,000 train rows.
  - TF-IDF (1–2-grams) + logistic regression on 4,000 train rows.

```bash
source /workspace/tools/setup.sh
PY=/workspace/tools/mlenv/bin/python          # venv with torch/transformers/sklearn/pandas
$PY scripts/prep.py
MODE=single $PY scripts/run_024_grok.py; MODE=batch $PY scripts/run_024_grok.py
MODEL=<path to SmolLM2-360M-Instruct> $PY scripts/run_024_local.py
$PY scripts/analyse_024.py                   # -> results/summary.csv, results/summary.json, chart.png
```

## Results (200 items)

| method | accuracy | cost per 1K items (API-equiv.) | time per 1K items |
|---|---|---|---|
| Grok CLI, 1 item per call (200 calls) | **90.5%** | $3.20 | 5,893 s sequential latency (198 s wall at 6 parallel for 200) |
| Grok CLI, 20 items per call (10 calls) | 89.0% | **$0.39** | 1,898 s sequential (83 s wall at 6 parallel for 200) |
| SmolLM2-360M, 4-shot prompt | 29.0% | $0 | 2,481 s (CPU) |
| SmolLM2-360M frozen + LR probe (1K train) | 89.0% | $0 | 422 s (CPU) + 475 s one-off training |
| TF-IDF + LR (4K train) | 87.5% | $0 | 0.07 s (CPU) + 1.5 s training |

- Grok single call: ~5,075 input tokens per call (almost all CLI overhead), 184 output tokens, of which 183 were reasoning. Total $0.640 for 200 items.
- Grok batch call: ~6,114 input tokens and 2,050 output tokens per call (1,962 reasoning). Total $0.078 for 200 items.
- 0 failed calls and 0 unparseable answers in either Grok mode.
- Per-class accuracy (World/Sports/Business/Sci-Tech): Grok single 90/100/90/82, Grok batch 92/96/86/82, probe 90/92/88/86, TF-IDF 90/96/80/84. Everyone finds Sci/Tech vs Business the hard pair.
- The 4-shot SmolLM2 predicted Sci/Tech for 188 of 200 items.

Chart: `chart.png`. Raw call logs: `results/grok_calls_*.csv` (exact usage and `total_cost_usd` from the CLI's JSON). Predictions: `results/grok_preds_*.json`, `results/local_results.json`.

## Caveats
- 200 items: one item is 0.5 points, so the 1.5–3 point gaps between Grok, the probe and TF-IDF are within noise.
- Costs are **API-equivalent estimates** from the CLI's `total_cost_usd`. On SuperGrok the real constraint is the usage window, not dollars. Local "$0" ignores hardware and electricity.
- AG News is old and public; Grok has likely seen it, and AG News labels themselves are noisy (many Business/Sci-Tech items are genuinely ambiguous), which caps everyone near 90%.
- The 4-shot result reflects first-token scoring with a 360M model and a long few-shot prompt; a better prompt or calibration might help, but the collapse to one label is typical of models this small.
- The trained baselines need labelled data (1K–4K rows). Grok needed none.
- Local timings are from a shared 2-core sandbox; treat them as rough.
