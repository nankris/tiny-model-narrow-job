import json, os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TE = json.load(open(f"{D}/results/test200.json")); gold = [e["label"] for e in TE]
rows = []
for mode, desc in [("single", "Grok CLI, 1 call per item"), ("batch", "Grok CLI, 20 items per call")]:
    p = json.load(open(f"{D}/results/grok_preds_{mode}.json")); c = pd.read_csv(f"{D}/results/grok_calls_{mode}.csv")
    ts = pd.to_datetime(c.ts_ist)
    rows.append(dict(method=desc, acc=np.mean([o["pred"] == o["gold"] for o in p]), n=len(p), calls=len(c),
                     usd_per_1k=c.cost_usd.sum() / len(p) * 1000, sec_per_item_seq=c.latency_s.sum() / len(p),
                     wall_s_total=(ts.max() - ts.min()).total_seconds() + c.latency_s.iloc[0], parallel=6,
                     in_tok_per_call=(c.input_tokens + c.cache_read_input_tokens).mean(), out_tok_per_call=c.output_tokens.mean(),
                     reasoning_tok_per_call=c.reasoning_tokens.mean()))
L = json.load(open(f"{D}/results/local_results.json"))
for k, desc in [("smollm2_360m_4shot", "SmolLM2-360M-Instruct, 4-shot (CPU)"), ("smollm2_360m_probe_1000", "SmolLM2-360M frozen + LR probe, 1K train (CPU)"),
                ("tfidf_lr_4000", "TF-IDF + logistic regression, 4K train (CPU)")]:
    r = L[k]; rows.append(dict(method=desc, acc=r["acc"], n=200, calls=0, usd_per_1k=0.0, sec_per_item_seq=r["infer_wall_s"] / 200,
                               wall_s_total=r["infer_wall_s"], parallel=1, train_s=r.get("train_s")))
t = pd.DataFrame(rows); t["sec_per_1k_items"] = t.sec_per_item_seq * 1000
t.to_csv(f"{D}/results/summary.csv", index=False); print(t.round(4).to_string())
# per-class accuracy
pc = {}
for k in ["smollm2_360m_4shot", "smollm2_360m_probe_1000", "tfidf_lr_4000"]:
    pc[k] = {l: float(np.mean([p == g for p, g in zip(L[k]["preds"], gold) if g == l])) for l in sorted(set(gold))}
for mode in ["single", "batch"]:
    p = json.load(open(f"{D}/results/grok_preds_{mode}.json")); pc["grok_" + mode] = {l: float(np.mean([o["pred"] == o["gold"] for o in p if o["gold"] == l])) for l in sorted(set(gold))}
json.dump(dict(table=t.round(5).to_dict(orient="records"), per_class=pc), open(f"{D}/results/summary.json", "w"), indent=1)
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
short = ["Grok\n1/call", "Grok\n20/call", "SmolLM2\n4-shot", "SmolLM2\n+LR probe", "TF-IDF\n+LR"]
cols = ["#4C72B0", "#6C8EBF", "#DD8452", "#E8A87C", "#55A868"]
a.bar(short, t.acc * 100, color=cols); a.set_ylim(0, 105); a.set_ylabel("accuracy on 200 AG News test items (%)"); a.set_title("Accuracy")
for i, v in enumerate(t.acc): a.text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center")
b.bar(short, t.usd_per_1k, color=cols); b.set_ylabel("USD per 1,000 items (API-equivalent)"); b.set_title("Cost per 1K items (local = $0 marginal)")
for i, v in enumerate(t.usd_per_1k): b.text(i, v + 0.03, f"${v:.2f}", ha="center")
fig.suptitle("News-topic classification: tiny local models vs Grok Build CLI (grok-4.7-build)")
plt.tight_layout(); plt.savefig(f"{D}/chart.png", dpi=130)
