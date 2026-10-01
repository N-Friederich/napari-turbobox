# Final benchmark results

## Validation (expected runs, configurations and protocol of every stage; one code state for all)

- main: k = {'modern': 5, 'legacy': 5}; 50 drags per configuration, pacing spin; OK
- large_n: k = {'modern': 5, 'legacy': 5}; 50 drags per configuration, pacing spin; OK
- sleep_pacing: k = {'modern': 3, 'legacy': 3}; 10 drags per configuration, pacing sleep; OK
- sleep_pacing_spin: k = {'modern': 3, 'legacy': 3}; 10 drags per configuration, pacing spin; OK
- tutorial_full: k = {'modern': 0, 'legacy': 3}; 10 drags per configuration, pacing spin; OK
- memory (2 tools, 9 N; store and index at 3 N) and image size (5 sizes): complete; OK
- every run: code state a8ad8214db5e at start and end, environment freeze 5d3c862b72f5 (drag runs), AC power, nice 0

Median of the k per-run medians, ms [bootstrap 95 % interval; with k <= 5 runs it equals the range of the run medians]. p95/p99: median over runs. Speed-up: ratio of the two medians [bootstrap interval]. Interactive: median, p95, p99 <= 33 ms. k per tool is given in each section.

## Main run, panel a (XY editable + XZ/YZ + 3D); k = [5] (TurboBox), [5] (napari-bbox)

| N | TurboBox | p95 / p99 | 3D update | napari-bbox per-box | p95 / p99 | 3D update | speed-up [CI] |
|---:|---|---|---|---|---|---|---|
| 10 | 1.6 [1.6–1.7] | 1.8 / 2.4 | 2.5 | 3.9 [3.9–3.9] | 4.2 / 4.7 | 9.7 | 2.4× [2.4–2.4] |
| 50 | 1.7 [1.7–1.7] | 1.8 / 2.2 | 4.7 | 4.5 [4.5–4.6] | 4.8 / 5.4 | 19.4 | 2.7× [2.7–2.8] |
| 100 | 1.7 [1.7–1.7] | 1.8 / 2.1 | 7.6 | 5.3 [5.2–5.3] | 5.6 / 6.2 | 31.7 | 3.2× [3.1–3.2] |
| 200 | 1.7 [1.7–1.7] | 1.8 / 2.1 | 8.4 | 6.7 [6.7–6.8] | 7.1 / 7.8 | 45.7 | 4.0× [4.0–4.1] |
| 500 | 1.7 [1.7–1.7] | 1.9 / 2.2 | 9.2 | 10.9 [10.8–11.0] | 11.6 / 12.5 | 78.4 | 6.5× [6.4–6.5] |
| 1000 | 1.7 [1.7–1.7] | 1.9 / 2.3 | 10.5 | 18.0 [17.8–18.2] | 19.0 / 20.8 | 132.1 | 10.5× [10.4–10.6] |

## Main run, panel b (V identical XY views, N = 100, no 3D); k = [5] / [5]

| V | TurboBox | p95 / p99 | napari-bbox per-box | p95 / p99 | ratio of medians [CI] |
|---:|---|---|---|---|---|
| 1 | 0.7 [0.7–0.7] | 0.8 / 1.1 | 1.4 [1.4–1.4] | 1.6 / 2.0 | 2.0× [2.0–2.1] |
| 2 | 1.2 [1.2–1.2] | 1.4 / 1.8 | 3.4 [3.4–3.4] | 3.6 / 4.1 | 2.8× [2.8–2.9] |
| 4 | 2.2 [2.2–2.2] | 2.4 / 2.9 | 7.3 [7.2–7.3] | 7.7 / 8.3 | 3.4× [3.3–3.4] |
| 8 | 4.1 [4.1–4.1] | 4.5 / 5.3 | 15.1 [15.0–15.1] | 15.6 / 16.6 | 3.7× [3.6–3.7] |

## Large-N run, panel a; k = [5] / [5]

| N | TurboBox | p95 / p99 | 3D update | napari-bbox per-box | p95 / p99 | 3D update | ratio of medians [CI] |
|---:|---|---|---|---|---|---|---|
| 2000 | 1.7 [1.7–1.8] | 1.9 / 2.5 | 11.3 | 32.4 [31.6–32.5] | 33.5 / 35.1 | 232.4 | 18.7× [18.2–18.9] |
| 5000 | 1.8 [1.8–1.8] | 2.0 / 3.1 | 12.9 | 79.2 [77.3–79.4] | 81.1 / 83.7 | 542.1 | 43.2× [42.1–43.5] |
| 10000 | 2.0 [2.0–2.0] | 2.3 / 4.4 | 15.8 | 158.7 [155.6–159.1] | 163.5 / 170.9 | 1057.9 | 79.1× [77.5–80.2] |

## Full-reassignment baseline (napari's multiple-viewer example pattern); k = [3]

| N | napari-bbox full | p95 / p99 | 3D update | TurboBox (main) | ratio of medians |
|---:|---|---|---|---|---|
| 10 | 23.1 [23.1–23.2] | 23.7 / 24.2 | 14.4 | 1.6 | 14.0× |
| 50 | 62.7 [62.6–62.8] | 63.9 / 64.7 | 54.5 | 1.7 | 37.7× |
| 100 | 112.2 [112.0–112.4] | 114.7 / 116.9 | 104.5 | 1.7 | 67.3× |
| 200 | 210.5 [210.3–211.7] | 213.1 / 215.2 | 173.2 | 1.7 | 125.7× |
| 500 | 472.5 [471.7–473.4] | 478.4 / 486.8 | 380.8 | 1.7 | 279.9× |
| 1000 | 908.6 [907.8–909.3] | 925.2 / 975.0 | 727.9 | 1.7 | 531.4× |

Busy-waiting reference: matched busy-waiting control (sleep_pacing_spin, same drags). A configuration whose busy-waiting median exceeds 33.3 ms never waits between moves, so its sleep row is a drift control, not a pacing comparison.

## Pacing sensitivity: idle waiting (sleep) vs busy-waiting (spin); k = [3] / [3] (10 drags per run; busy-waiting reference: matched busy-waiting control (sleep_pacing_spin, same drags))

| panel | N | V | TurboBox sleep | TurboBox spin | napari-bbox sleep | napari-bbox spin | speed-up sleep [CI] | speed-up spin | TB p99 sleep / spin | BB p99 sleep / spin |
|---|---:|---:|---|---|---|---|---|---|---|---|
| a | 10 | 3+3D | 8.1 [8.1–8.1] | 1.6 | 8.9 [8.8–9.0] | 3.9 | 1.1× [1.1–1.1] | 2.4× | 11.5 / 2.3 | 10.8 / 4.6 |
| a | 100 | 3+3D | 8.2 [8.1–8.2] | 1.7 | 9.8 [9.7–9.9] | 5.2 | 1.2× [1.2–1.2] | 3.2× | 11.0 / 2.0 | 11.0 / 6.1 |
| a | 1000 | 3+3D | 8.5 [8.4–8.6] | 1.7 | 19.8 [19.8–20.1] | 17.7 | 2.3× [2.3–2.4] | 10.4× | 11.1 / 2.3 | 21.6 / 20.5 |
| a | 10000 | 3+3D | 8.0 [7.8–8.2] | 2.0 | 155.7 [155.2–158.9] | 159.1 | 19.4× [18.8–20.5] | 79.9× | 13.3 / 4.5 | 167.9 / 174.7 |
| b | 100 | 1 | 5.3 [5.3–5.4] | 0.7 | 6.8 [6.7–6.8] | 1.4 | 1.3× [1.3–1.3] | 2.1× | 7.5 / 1.1 | 9.8 / 2.0 |
| b | 100 | 2 | 7.1 [7.1–7.2] | 1.2 | 8.5 [8.4–8.6] | 3.4 | 1.2× [1.2–1.2] | 2.8× | 9.9 / 1.5 | 13.1 / 4.1 |
| b | 100 | 4 | 8.9 [8.7–9.2] | 2.2 | 11.2 [11.1–11.2] | 7.2 | 1.3× [1.2–1.3] | 3.4× | 11.8 / 2.6 | 12.3 / 8.1 |
| b | 100 | 8 | 9.2 [9.2–9.2] | 4.1 | 17.8 [17.8–17.9] | 15.0 | 1.9× [1.9–1.9] | 3.7× | 10.7 / 4.7 | 18.9 / 16.4 |

## Interactive limits (median, p95, p99 <= 33 ms), panel a

- TurboBox: interactive at N = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
- napari-bbox per-box: interactive at N = [10, 50, 100, 200, 500, 1000]

## Memory (tracemalloc, one hidden 2D viewer, napari 0.7.0; MiB)

| N | TurboBox | napari-bbox | ratio |
|---:|---|---|---|
| 10 | 0.07 | 0.12 | 1.7× |
| 50 | 0.12 | 0.34 | 2.8× |
| 100 | 0.19 | 0.55 | 2.9× |
| 200 | 0.35 | 0.96 | 2.8× |
| 500 | 0.79 | 2.19 | 2.8× |
| 1000 | 1.56 | 4.24 | 2.7× |
| 2000 | 3.01 | 8.32 | 2.8× |
| 5000 | 6.85 | 20.19 | 2.9× |
| 10000 | 13.66 | 39.99 | 2.9× |

Store + one spatial index (TurboBox), bytes: N=100: 16800, N=1000: 168000, N=10000: 1680000

## Image size (TurboBox, N = 100, 4 XY/YZ/XZ views, no 3D)

| image px/side | median ms | p95 ms |
|---:|---|---|
| 128 | 2.1 | 2.4 |
| 256 | 2.1 | 2.3 |
| 512 | 2.1 | 2.3 |
| 1024 | 2.1 | 2.3 |
| 2048 | 2.1 | 2.3 |

## Wall time per stage (first start to last finish)

- large_n: 13:00–14:49 (1.83 h)
- main: 14:50–17:45 (2.92 h)
- sleep_pacing: 17:45–18:28 (0.72 h)
- sleep_pacing_spin: 17:53–18:37 (0.72 h)
- tutorial_full: 18:41–19:42 (1.03 h)
