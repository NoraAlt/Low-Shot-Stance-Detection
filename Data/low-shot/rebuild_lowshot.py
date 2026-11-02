"""Rebuild the low-shot training files used in the paper.

Each low-shot training file = Data/train_prompted.json with the held-out stance
entries removed (same filter as code/utils/Low_shot.py), plus the exact support
set that was sampled for that run (Data/low-shot/support_sets/*.jsonl).
The support sets were sampled without a fixed seed, so they are stored as files.

Usage (from the repo root):
    python Data/low-shot/rebuild_lowshot.py            # writes Data/low-shot/train_3N_*.json
"""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "..", "train_prompted.json")
STANCE = ["Mawqif_stance_train", "AraCovidVac_traindev"]
TARGET = {"Women": "regarding: تمكين المرأة?",
          "Digital": "regarding: التحول الرقمي?",
          "Covid": "regarding: لقاح كورونا?"}

# run name -> (datasets the held-out filter applies to, target prompt filter or None for all stance)
RUNS = {}
for k in (0, 5, 10, 20):
    RUNS[f"train_3N_{k}K_All"] = (STANCE, None)
    for t in TARGET:
        RUNS[f"train_3N_{k}K_{t}"] = (STANCE, TARGET[t])
RUNS["train_3N_0K_Covid_withAracovid"] = (["Mawqif_stance_train"], TARGET["Covid"])


def removed(entry, datasets, prompt_filter):
    if entry["dataset"] not in datasets:
        return False
    return prompt_filter is None or prompt_filter in entry["prompt"]


def build(name, base):
    datasets, pf = RUNS[name]
    kept = [e for e in base if not removed(e, datasets, pf)]
    sup_path = os.path.join(ROOT, "support_sets", name + ".jsonl")
    support = [json.loads(l) for l in open(sup_path, encoding="utf-8")] if os.path.exists(sup_path) else []
    return kept + support


if __name__ == "__main__":
    base = [json.loads(l) for l in open(BASE, encoding="utf-8")]
    for name in RUNS:
        rows = build(name, base)
        with open(os.path.join(ROOT, name + ".json"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}.json: {len(rows)} rows")
