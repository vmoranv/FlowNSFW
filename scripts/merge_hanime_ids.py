import json, sys
from pathlib import Path
from collections import Counter

out = Path("D:/cumhub/flow-nsfw/datasets/hanime_all_ids.json")

data = json.loads(sys.stdin.read())
seen = {}
domain_map = {"裏番":"anime2d","3DCG":"render3d","2.5D":"semi2_5d","MMD":"render3d"}

for key, ids in data.items():
    genre = key.rsplit("_p", 1)[0]
    domain = domain_map.get(genre, "semi2_5d")
    for vid in ids:
        if vid not in seen:
            seen[vid] = {"genre": genre, "domain": domain}

dc = Counter(v["domain"] for v in seen.values())
anime2d = [k for k,v in seen.items() if v["domain"]=="anime2d"][:100]
render3d = [k for k,v in seen.items() if v["domain"]=="render3d"][:100]
semi = [k for k,v in seen.items() if v["domain"]=="semi2_5d"][:100]

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"all": seen, "queue": {"anime2d": anime2d, "render3d": render3d, "semi2_5d": semi}}, ensure_ascii=False))
print(f"Total unique: {len(seen)} | domains: {dict(dc)}")
print(f"Download queue: anime2d={len(anime2d)} render3d={len(render3d)} semi2_5d={len(semi)}")
