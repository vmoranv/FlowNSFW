"""Full evaluation: all 224 training videos + 10 fresh NSFW videos."""
import json, sys, torch, os
sys.path.insert(0, "src")

from flow_nsfw import FlowNSFW
from flow_nsfw.data import VideoClipDataset
from torch.utils.data import ConcatDataset

CKPT = "final.pt"
MANIFEST = "datasets/manifest_v4_clean_wsl.json"
FRESH = "test_fresh/fresh_manifest.json"
RESOLUTIONS = [256, 320, 480, 640]

ck = torch.load(CKPT, map_location="cuda")
model = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
    temporal_backend="mamba", d_state=16, ssm_expand=2, sparse_detect=True,
).cuda()
model.load_state_dict(ck["model"])
model.eval()
print(f"Model: step={ck.get('step','?')}\n")

# ====== 1. FULL ORIGINAL DATASET ======
print("=" * 70)
print("  FULL DATASET EVAL (224 videos)")
print("=" * 70)

for res in RESOLUTIONS:
    ds_train = VideoClipDataset(MANIFEST, clip_len=4, resolution=(res, res), split="train")
    ds_val = VideoClipDataset(MANIFEST, clip_len=4, resolution=(res, res), split="val")
    ds_all = ConcatDataset([ds_train, ds_val])

    nsfw_ok = sfw_ok = nsfw_tot = sfw_tot = 0
    fp_vids, fn_vids = [], []
    fn_conf = []

    for i in range(len(ds_all)):
        s = ds_all[i]
        label = s["video_label"].item() if hasattr(s["video_label"], "item") else s["video_label"]
        vid = s.get("video_id", f"vid_{i}")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(s["frames"].unsqueeze(0).cuda())
        pred = o["video_cls"].argmax(-1).item()
        probs = torch.softmax(o["video_cls"], dim=-1)

        if label == 1:
            nsfw_tot += 1
            if pred == 1: nsfw_ok += 1
            else: fn_vids.append(vid); fn_conf.append(f"{probs[0,1].item():.4f}")
        else:
            sfw_tot += 1
            if pred == 0: sfw_ok += 1
            else: fp_vids.append(vid)

    acc = (nsfw_ok + sfw_ok) / max(1, nsfw_tot + sfw_tot) * 100
    rec = nsfw_ok / max(1, nsfw_tot) * 100
    prec = nsfw_ok / max(1, nsfw_ok + len(fp_vids)) * 100
    status = "🎉" if acc >= 95 else "✅" if acc >= 90 else "⚠️"
    print(f"  {status} {res}p: Acc={acc:.1f}% Prec={prec:.1f}% Rec={rec:.1f}% "
          f"NSFW={nsfw_ok}/{nsfw_tot} SFW={sfw_ok}/{sfw_tot} FP={len(fp_vids)} FN={len(fn_vids)}")

    if fn_vids:
        print(f"      FN: " + ", ".join(fn_vids[:8]))
        fn_conf_str = " ".join(f"({c})" for c in fn_conf[:8])
        print(f"      conf: {fn_conf_str}")

    if fp_vids:
        print(f"      FP: " + ", ".join(fp_vids[:5]))

# ====== 2. FRESH NSFW VIDEOS ======
if os.path.exists(FRESH):
    fresh_m = json.load(open(FRESH))
    print(f"\n{'='*70}")
    print(f"  FRESH NSFW TEST ({len(fresh_m)} videos — NOT in training data)")
    print(f"{'='*70}")

    # Convert Windows paths to WSL if needed
    from torch.utils.data import Dataset

    class FreshDataset(Dataset):
        def __init__(self, manifest_path, resolution):
            import json
            raw = json.load(open(manifest_path))
            # Fix paths for WSL
            for v in raw:
                v["frames"] = [f.replace("D:\\", "/mnt/d/").replace("\\", "/") for f in v["frames"]]
            # Write temp manifest
            tmp_path = "/tmp/fresh_manifest_wsl.json"
            json.dump(raw, open(tmp_path, "w"))
            self.ds = VideoClipDataset(tmp_path, clip_len=4, resolution=(resolution, resolution), split="test")
            self.raw = raw
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            s = self.ds[i]
            s["_idx"] = i
            return s

    for res in [320, 480]:
        fd = FreshDataset(FRESH, res)
        nsfw_hit = 0
        results = []
        for i in range(len(fd)):
            s = fd[i]
            idx = s.get("_idx", i)
            vid = fresh_m[idx].get("video_id", f"?")
            frames_path = fresh_m[idx].get("frames", ["?"])[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(s["frames"].unsqueeze(0).cuda())
            pred = o["video_cls"].argmax(-1).item()
            probs = torch.softmax(o["video_cls"], dim=-1)
            if pred == 1: nsfw_hit += 1
            status = "✅" if pred == 1 else "❌ MISS"
            results.append((status, vid, probs[0,1].item(), frames_path))

        acc = nsfw_hit / max(1, len(fd)) * 100
        print(f"\n  @ {res}p: {nsfw_hit}/{len(fd)} detected = {acc:.0f}%")
        for status, vid, conf, fpath in results:
            print(f"    {status} {vid}  conf={conf:.3f}")
            print(f"         {fpath}")

print(f"\n{'='*70}")
print("  EVALUATION COMPLETE")
