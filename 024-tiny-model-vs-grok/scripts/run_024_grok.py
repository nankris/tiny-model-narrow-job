# Grok on 200 AG News items. MODE=single: one call per item. MODE=batch: 20 items per call (10 calls).
import json, os, sys, re, tempfile, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(__file__))
from grok_log import run_grok, QuotaHit
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = json.load(open(f"{D}/results/test200.json")); MODE = os.environ.get("MODE", "single")
LAB = ["World", "Sports", "Business", "Sci/Tech"]
LEAN = ["--system-prompt-override", "You are a news topic classifier.", "--disable-web-search", "--no-subagents", "--tools", "read_file"]
CSV = f"{D}/results/grok_calls_{MODE}.csv"; OUT = f"{D}/results/grok_preds_{MODE}.json"
def norm(s):
    s = s.strip().lower()
    for l in LAB:
        if l.lower() in s or (l == "Sci/Tech" and ("sci" in s or "tech" in s)): return l
    return None
def single(e):
    p = f"Classify this news item into exactly one of: World, Sports, Business, Sci/Tech. Reply with the label only.\n\n{e['text']}"
    r = run_grok(p, f"id{e['id']}", CSV, "024-single", 1, cwd=tempfile.mkdtemp(), extra=LEAN)
    return [{"id": e["id"], "gold": e["label"], "pred": norm(r["text"]), "raw": r["text"][:80]}]
def batch(chunk, k):
    items = "\n".join(f"{j+1}. {e['text']}" for j, e in enumerate(chunk))
    p = ("Classify each numbered news item into exactly one of: World, Sports, Business, Sci/Tech. "
         "Reply with one line per item in the form '<number>: <label>' and nothing else.\n\n" + items)
    r = run_grok(p, f"batch{k}", CSV, "024-batch", 1, cwd=tempfile.mkdtemp(), extra=LEAN)
    got = {int(m.group(1)): m.group(2) for m in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*(.+)$", r["text"], re.M)}
    return [{"id": e["id"], "gold": e["label"], "pred": norm(got.get(j + 1, "")), "raw": got.get(j + 1, "")} for j, e in enumerate(chunk)]
done = {o["id"]: o for o in (json.load(open(OUT)) if os.path.exists(OUT) else [])}
todo = [e for e in EX if e["id"] not in done]
jobs = [(single, (e,)) for e in todo] if MODE == "single" else [(batch, (EX[i:i+20], i // 20)) for i in range(0, 200, 20) if any(e["id"] not in done for e in EX[i:i+20])]
hit = False
with cf.ThreadPoolExecutor(6) as ex:
    futs = [ex.submit(f, *a) for f, a in jobs]
    for f in cf.as_completed(futs):
        try:
            for o in f.result(): done[o["id"]] = o
        except QuotaHit as e:
            hit = True; print("QUOTA HIT", e)
json.dump(sorted(done.values(), key=lambda o: o["id"]), open(OUT, "w"), indent=1)
print(MODE, "done", len(done), "acc", sum(o["pred"] == o["gold"] for o in done.values()) / max(1, len(done)))
sys.exit(3 if hit else 0)
