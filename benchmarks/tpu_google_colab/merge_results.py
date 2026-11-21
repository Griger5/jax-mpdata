from pathlib import Path
import json

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR.parent / "results"
DEVICES = ("cpu", "gpu", "tpu")

if __name__ == "__main__":
    paths = {dev: RESULTS_DIR / f"tpu_google_colab_{dev}.json" for dev in DEVICES}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {missing} in {RESULTS_DIR}")

    merged = {}
    for path in paths.values():
        part = json.loads(path.read_text())
        print(f"{path.name}: {len(part)} grids")
        for name, entry in part.items():
            target = merged.setdefault(name, {"metadata": entry["metadata"], "data": {}})
            target["data"].update(entry["data"])

    out = RESULTS_DIR / "tpu_google_colab.json"
    with open(out, "w", encoding="UTF-8") as f:
        json.dump(merged, f, sort_keys=True, indent=4)

    columns = sorted({k for e in merged.values() for k in e["data"]})
    for col in columns:
        kinds = sorted({e["data"][col]["device"] for e in merged.values() if col in e["data"]})
        print(col, "->", kinds)

    print()
    print("min seconds".ljust(26), " ".join(c.rjust(16) for c in columns))
    for name in sorted(merged, key=lambda n: (merged[n]["metadata"]["size_x"], merged[n]["metadata"]["n_iters"])):
        row = merged[name]["data"]
        print(name.ljust(26), " ".join(
            format(min(row[c]["times"]), ".6g").rjust(16) if c in row else "-".rjust(16)
            for c in columns))

    print(f"\nWrote {out}")