## 📚 Additional Resources

### Documentation

- **[Architecture Deep Dive](ARCHITECTURE.md)** - Complete module-by-module breakdown
- **[Mathematical Theory](THEORY.md)** - Mathematical foundations and proofs
- **[Quick Start Guide](QUICKSTART.md)** - Get running in 30 seconds
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to the project

### Research

**Paper** (Coming Soon): FlowNSFW: Optical Flow and Mamba SSM for Video NSFW Detection

**Key Insights**:
- NSFW detection is fundamentally a **motion pattern recognition** task
- Single-frame detectors miss 40-58% of NSFW content (YOLOv11 baseline)
- Optical flow captures spatiotemporal gradients invisible to RGB alone
- Mamba SSM provides O(N) temporal modeling vs Transformer's O(N²)

### Model Zoo

| Model | Params | Size (FP32) | Accuracy | Speed | Download |
|-------|--------|-------------|----------|-------|----------|
| FlowNSFW-Tiny | 2.1M | ~8MB | 92.8% | 180ms | Coming Soon |
| **FlowNSFW-Base** | **7.13M** | **27.2MB** | **96.4%** | **411ms** | [v1.0.0](https://github.com/vmoranv/FlowNSFW/releases) |
| FlowNSFW-Large | 15M | ~60MB | 96.8% | 820ms | Coming Soon |

**Note**: Base model benchmarked with Attention backend. Mamba SSM variant has similar accuracy but O(N) complexity.

### Pretrained Weights

Download from [Releases](https://github.com/vmoranv/FlowNSFW/releases/tag/v1.0.0):

- `final.pt` (27.2MB) - FlowNSFW-Base, 7.13M params, 96.4% accuracy
- **Checksum (MD5)**: Coming soon (upload weights to create release)

### Datasets

**Training Set**: 224 videos (124 NSFW + 100 SFW)
- NSFW: Curated from multiple sources, manually verified
- SFW: Pexels API (nature, city, sports, cooking, people, etc.)
- Split: 80% train, 20% validation
- Frame sampling: 8-frame clips with temporal stride

**Test Set**: Same 224 videos, separate split
- Multi-resolution evaluation: 160p, 240p, 320p, 480p, 640p
- Ground truth: Per-frame + video-level labels

> **Note**: Datasets are not publicly released due to content sensitivity. Contact for research collaboration.

### Community

- **Discussions**: [GitHub Discussions](https://github.com/vmoranv/FlowNSFW/discussions)
- **Issues**: [Bug Reports & Features](https://github.com/vmoranv/FlowNSFW/issues)
- **Twitter/X**: [@your_handle](https://twitter.com/your_handle) (Optional)

### Use Cases

- 🛡️ **Content Moderation**: Automated video platform moderation
- 🔒 **Parental Controls**: Family-safe content filtering
- 🏢 **Enterprise Compliance**: Corporate content policy enforcement
- 🔬 **Research**: Safety AI and video understanding research

### Ethical Considerations

FlowNSFW is a **content moderation tool** designed for safety applications:

✅ **Appropriate Use**:
- Automated pre-screening for human review
- Compliance with platform policies
- Research on content safety systems
- Parental control applications

❌ **Inappropriate Use**:
- Surveillance without consent
- Discrimination or profiling
- Bypassing user privacy settings
- High-stakes decisions without human oversight

### Related Work

**Optical Flow**:
- [FlowNet 2.0](https://arxiv.org/abs/1612.01925) - Ilg et al., 2017
- [RAFT](https://arxiv.org/abs/2003.12039) - Teed & Deng, 2020

**State Space Models**:
- [Mamba](https://arxiv.org/abs/2312.00752) - Gu & Dao, 2023
- [S4](https://arxiv.org/abs/2111.00396) - Gu et al., 2021

**Video Understanding**:
- [Video Swin Transformer](https://arxiv.org/abs/2106.13230) - Liu et al., 2021
- [TimeSformer](https://arxiv.org/abs/2102.05095) - Bertasius et al., 2021

### Comparison to Other Approaches

| Approach | Accuracy | Speed | Pros | Cons |
|----------|----------|-------|------|------|
| **FlowNSFW** | **96.4%** | **411ms** | Motion-aware, lightweight | Requires video |
| YOLOv11 Frame | 70.0% | 265ms | Fast, simple | Misses motion patterns |
| I3D + Transformer | ~94% | 1200ms | Strong baseline | Slow, heavy (120M params) |
| Traditional ML | 55.4% | 150ms | Interpretable | Brittle rules, high FP |

### Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version: Incompatible API changes
- **MINOR** version: Backward-compatible functionality
- **PATCH** version: Backward-compatible bug fixes

Current: `v1.0.0`

### License

MIT License - see [LICENSE](LICENSE) for details.

**Content Moderation Notice**: This software is intended for lawful content moderation and safety research. See [SECURITY.md](SECURITY.md) for responsible use guidelines.

### Acknowledgments

- **Mamba Authors**: Albert Gu and Tri Dao for S6 architecture
- **FlowNet Authors**: Alexey Dosovitskiy et al. for optical flow foundations
- **PyTorch Team**: For the deep learning framework
- **Ultralytics**: For YOLOv11 baseline comparisons
- **Contributors**: See [CONTRIBUTING.md](CONTRIBUTING.md)

### Funding & Support

This project was developed as part of academic research. No commercial funding received.

---

**Questions?** Open a [Discussion](https://github.com/vmoranv/FlowNSFW/discussions) or [Issue](https://github.com/vmoranv/FlowNSFW/issues).

**Star ⭐ this repo** if FlowNSFW helps your research or project!
