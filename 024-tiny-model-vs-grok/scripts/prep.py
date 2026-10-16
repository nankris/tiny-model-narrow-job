# 200 examples (50 per class) from the public AG News test split, seed 0. Also save 4000 train rows for baselines.
import json, os, random
from datasets import load_dataset
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = ["World", "Sports", "Business", "Sci/Tech"]
te = load_dataset("fancyzhx/ag_news", split="test"); tr = load_dataset("fancyzhx/ag_news", split="train")
random.seed(0); idx = {c: [i for i, l in enumerate(te["label"]) if l == c] for c in range(4)}
pick = sorted(sum((random.sample(idx[c], 50) for c in range(4)), []))
ex = [{"id": i, "text": te[i]["text"], "label": LAB[te[i]["label"]]} for i in pick]
json.dump(ex, open(f"{D}/results/test200.json", "w"), indent=1)
tri = random.sample(range(len(tr)), 4000)
json.dump([{"text": tr[i]["text"], "label": LAB[tr[i]["label"]]} for i in tri], open(f"{D}/results/train4000.json", "w"))
print(len(ex), ex[0])
