# Academic Repository Structure

## ✅ Keep (Core)
- src/flow_nsfw/          # Source code
- scripts/                # Training/inference/evaluation scripts
- docs/                   # Documentation
- .github/                # GitHub templates and CI
- assets/                 # Figures for README

## ✅ Keep (Documentation)
- README.md
- LICENSE
- CITATION.cff
- CONTRIBUTING.md
- SECURITY.md
- CHANGELOG.md
- ARCHITECTURE.md
- THEORY.md
- QUICKSTART.md
- RESOURCES.md
- requirements.txt
- .gitignore

## ❌ Remove (Development Artifacts)
- delivery_package/       # Packaged version for deployment
- flow_nsfw_a10_package/  # A10 GPU specific build
- flow_nsfw_v10/          # Old version snapshot
- runs/                   # Training logs
- datasets/               # Data (not for public repo)
- output_boxes/           # Debug outputs
- test_fresh/             # Test artifacts
- baselines/              # Baseline model outputs
- configs/                # Local training configs
- build_v10.py            # Build scripts
- fix_*.py, fix_*.sh      # Hotfix scripts
- test_*.py               # Ad-hoc test scripts
- v10_patch/              # Patches
- *.tar.gz, *.zip         # Archives

## 📁 Ideal Academic Repo Structure
```
FlowNSFW/
├── .github/
│   ├── workflows/test.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── assets/
│   └── performance_comparison.png
├── docs/
│   └── SFW_COLLECTION_GUIDE.md
├── scripts/
│   ├── train.py
│   ├── infer.py
│   ├── eval_multi_res.py
│   ├── bench_full.py
│   └── generate_figures.py
├── src/flow_nsfw/
│   ├── __init__.py
│   ├── model.py
│   ├── flow_net.py
│   ├── temporal_sparse.py
│   ├── ssm_backend.py
│   ├── detection_head.py
│   ├── encoder_unet.py
│   ├── losses.py
│   ├── data.py
│   ├── balanced_sampler.py
│   └── utils.py
├── .gitignore
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── SECURITY.md
├── CHANGELOG.md
├── ARCHITECTURE.md
├── THEORY.md
├── QUICKSTART.md
├── RESOURCES.md
└── requirements.txt
```

Total: ~40 files, ~20k lines of code + docs
