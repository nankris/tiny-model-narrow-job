# Local baselines on the same 200 AG News test items, CPU only (2 threads).
#  A) SmolLM2-360M-Instruct, 4-shot prompt, label chosen by next-token logits over the 4 label words (1 forward pass / item)
#  B) SmolLM2-360M-Instruct as a frozen encoder: mean-pooled hidden states -> logistic regression trained on 1,000 train rows
#  C) TF-IDF + logistic regression trained on 4,000 train rows (classic non-LLM baseline)
import json, os, time, torch, numpy as np, resource
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("MODEL", "/workspace/projects/4e0e7b68-80db-40d3-a6d5-b21cf99d82b6/work/SmolLM2-360M-Instruct")  # HuggingFaceTB/SmolLM2-360M-Instruct
LAB = ["World", "Sports", "Business", "Sci/Tech"]
TE = json.load(open(f"{D}/results/test200.json")); TR = json.load(open(f"{D}/results/train4000.json"))
torch.set_num_threads(2); res = {}
cpu = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime

# C) TF-IDF + LR
t0, c0 = time.time(), cpu()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); Xtr = vec.fit_transform([e["text"] for e in TR])
clf = LogisticRegression(max_iter=2000, C=10).fit(Xtr, [e["label"] for e in TR]); t_train = time.time() - t0
t1, c1 = time.time(), cpu(); pred = clf.predict(vec.transform([e["text"] for e in TE]))
res["tfidf_lr_4000"] = dict(acc=float(np.mean([p == e["label"] for p, e in zip(pred, TE)])), train_s=round(t_train, 2),
                            infer_wall_s=round(time.time() - t1, 3), infer_cpu_s=round(cpu() - c1, 3), preds=list(pred))
print("tfidf", res["tfidf_lr_4000"]["acc"], flush=True)

tok = AutoTokenizer.from_pretrained(MODEL); m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

# A) 4-shot, next-token scoring
shots = {}
for e in TR:
    if e["label"] not in shots and len(e["text"]) < 300: shots[e["label"]] = e["text"]
    if len(shots) == 4: break
sysmsg = "Classify the news item into one topic: World, Sports, Business, or Sci/Tech. Answer with the topic only."
def chat(text):
    msgs = [{"role": "system", "content": sysmsg}]
    for l in LAB: msgs += [{"role": "user", "content": shots[l]}, {"role": "assistant", "content": l}]
    msgs.append({"role": "user", "content": text})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
first = [tok(l, add_special_tokens=False)["input_ids"][0] for l in LAB]
assert len(set(first)) == 4, first
t0, c0 = time.time(), cpu(); preds = []
with torch.no_grad():
    for e in TE:
        ids = tok(chat(e["text"]), return_tensors="pt")
        lg = m(**ids).logits[0, -1, first]; preds.append(LAB[int(lg.argmax())])
res["smollm2_360m_4shot"] = dict(acc=float(np.mean([p == e["label"] for p, e in zip(preds, TE)])), infer_wall_s=round(time.time() - t0, 2),
                                 infer_cpu_s=round(cpu() - c0, 2), preds=preds)
print("4shot", res["smollm2_360m_4shot"]["acc"], res["smollm2_360m_4shot"]["infer_wall_s"], flush=True)

# B) frozen encoder + LR probe
def emb(texts, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128)
            h = m.model(**enc).last_hidden_state; msk = enc["attention_mask"].unsqueeze(-1)
            out.append(((h * msk).sum(1) / msk.sum(1)).numpy())
    return np.concatenate(out)
if tok.pad_token is None: tok.pad_token = tok.eos_token
t0 = time.time(); Etr = emb([e["text"] for e in TR[:1000]])
probe = LogisticRegression(max_iter=3000, C=1.0).fit(Etr, [e["label"] for e in TR[:1000]]); t_train = time.time() - t0
t1, c1 = time.time(), cpu(); pp = probe.predict(emb([e["text"] for e in TE]))
res["smollm2_360m_probe_1000"] = dict(acc=float(np.mean([p == e["label"] for p, e in zip(pp, TE)])), train_s=round(t_train, 1),
                                      infer_wall_s=round(time.time() - t1, 2), infer_cpu_s=round(cpu() - c1, 2), preds=list(pp))
print("probe", res["smollm2_360m_probe_1000"]["acc"], flush=True)
json.dump(res, open(f"{D}/results/local_results.json", "w"), indent=1)
