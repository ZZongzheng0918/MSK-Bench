# MSK-Bench

MSK-Bench: Benchmarking Full-Body Musculoskeletal Motor Control Across Tasks, Control Paradigms, and Physiological Metrics

Academic project page aligned with the supplied `MSK_Bench-4.pdf` manuscript.

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

- `index.html`: manuscript-aligned narrative, metric definitions, and Tables II–V.
- `static/images/`: paper figures and task posters. `main22.png` retains its existing filename but displays the 20 task curves in the revised Fig. 2.
- `static/js/video-gallery.js`: task families and policy video coverage.
- `static/videos/`: existing supplementary rollouts, preserved from the original site.

The public author list is retained from the existing project page; the supplied manuscript is anonymized. Paper results are point estimates. Focused studies must not be interpreted as a five-paradigm full-suite leaderboard, and EMG-envelope correlations measure phase-optimized waveform shape rather than absolute amplitude or timing.
