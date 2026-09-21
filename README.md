# Exploiting dynamic spatio-temporal correlations for origin-destination demand prediction

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.eswa.2025.130095-blue)](https://doi.org/10.1016/j.eswa.2025.130095)
[![Open access](https://img.shields.io/badge/open%20access-hybrid-brightgreen)](https://doi.org/10.1016/j.eswa.2025.130095)
[![Project page](https://img.shields.io/badge/project-page-blue)](https://codezx6.github.io/papers/dstcn.html)

Official implementation of **DSTCN** — *Exploiting dynamic spatio-temporal correlations for origin-destination demand prediction* (Expert Systems with Applications 2026). DSTCN (Dynamic Spatio-Temporal Correlation Network) forecasts origin-destination demand with three modules: Glstm2D for bidirectional origin/destination demand trends, Simformer for Transformer-based inter-regional similarity over the OD matrix, and FF-TM for temporal fusion; it beats state-of-the-art baselines on NYC-TOD2018, NYC-TOD2019, and HZMetro.

📄 Paper: https://doi.org/10.1016/j.eswa.2025.130095 · 🌐 Project page with abstract, FAQ and BibTeX: https://codezx6.github.io/papers/dstcn.html · 👤 Author: [Xu Zhang](https://codezx6.github.io)


A deep learning framework for metro Origin-Destination (OD) matrix prediction with temporal-spatial attention mechanisms and delayed flow completion strategies.

## Architecture

The framework implements a multi-granularity temporal encoding architecture combining:
- Recurrent graph convolutional units
- Dual-stream attention mechanisms  
- Bidirectional temporal encoding
- Adaptive flow completion modules


## Citation
If you use this work, please cite:

```bibtex
@article{gong2026dstcn,
  title        = {Exploiting dynamic spatio-temporal correlations for origin-destination demand prediction},
  author       = {Gong, Yongshun and Yu, Piao and Zhang, Xu and Zhang, Xinxin and Nie, Xiushan and Sun, Haoliang},
  journal      = {Expert Systems with Applications},
  year         = {2026},
  volume       = {299},
  pages        = {130095},
  doi          = {10.1016/j.eswa.2025.130095},
  issn         = {0957-4174},
  url          = {https://doi.org/10.1016/j.eswa.2025.130095}
}
```