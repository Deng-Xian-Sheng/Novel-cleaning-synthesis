import json

with open("dataset.state.json", "r") as f:
    data = json.load(f)

deletes = []

for v in data["done"].keys():
    status = data["done"][v]["status"]
    if status == "skipped_nontext_or_too_short":
        deletes.append(v)

for v in deletes:
    del(data["done"][v])

with open("dataset.state.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)