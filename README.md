# MSK-Bench

MSK-Bench: Benchmarking Full-Body Musculoskeletal Motor Control Across Tasks, Control Paradigms, and Physiological Metrics

Academic project page aligned with the supplied `MSK_Bench_RSS_workshop-3.pdf` manuscript.

- **22 tasks:** 6 stabilization, 6 locomotion, and 10 interaction tasks.
- **416-muscle model** for the complete task suite; the 700-muscle model is used only in the anatomy study.
- **5 control paradigms**, with a four-algorithm full-suite comparison and separately reported focused studies.
- **7 metric families:** success rate, cumulative reward, peak-efficiency steps, perturbation robustness, activation cost, joint smoothness, and EMG-envelope similarity.

## Preview

This is a static website with no build step or package installation:

```sh
python3 -m http.server 8000
```

Open http://localhost:8000. The paper is available at `static/papers/MSK-Bench.pdf`.

## Content sources

- `index.html`: manuscript-aligned narrative, metric definitions, and Tables I–V. Table I is an accessible HTML benchmark comparison. The Leaderboard section contains three detailed tables for all 22 tasks and four algorithms, preserving the supplied activation costs, 15 perturbation-sweep values, maximum steps, training-environment success rates, and smoothness values.
- `static/images/`: paper figures and task posters. Figures 1–10 are rendered from the supplied manuscript; `main22.png` contains all 22 task curves in Fig. 2.
- `static/js/video-gallery.js`: task families and policy video coverage.
- `static/videos/`: existing supplementary rollouts, preserved from the original site.

The nine authors, affiliation numbering, and BibTeX follow the supplied named manuscript. Running EMG results use 11 independent channels; stairs uses 12 matched muscles. The Paper button serves the complete 35-page manuscript, including its appendix. The only project buttons are Paper and [Code](https://github.com/ZZongzheng0918/MSK-Bench/tree/code). Paper results are point estimates. Focused studies must not be interpreted as a five-paradigm full-suite leaderboard, and EMG-envelope correlations measure phase-optimized waveform shape rather than absolute amplitude or timing.
