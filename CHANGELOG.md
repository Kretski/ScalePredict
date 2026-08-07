# Changelog

All notable changes to ScalePredict are documented here.

## [1.0.0] — 2026-08-07

### First public release

**Paper:** [10.5281/zenodo.21842461](https://doi.org/10.5281/zenodo.21842461)

#### Added
- `WTwinMonitor` — streaming W-Twin detector (W = Q·(D−T))
- `PowerLawBaseline` — ScalePredict scaling-law baseline predictor
- `run_ablation()` — comparative benchmark vs Threshold and CUSUM
- CLI: `scalepredict monitor log.csv` and `scalepredict demo`
- `pyproject.toml` — pip installable package

#### Key results (real nano-GPT training runs)
- Progressive drift: 9/9 detection (100%), mean delay 223 ± 11 steps
- False alarm rate: 0/30 on clean runs
- Abrupt failures: detected, CUSUM faster (+1 vs +5 steps)
- Threshold: 0 detections across all failure types

#### Known limitations
- Validated on nano-GPT (842K params) with synthetic byte-level text
- Power-law baseline assumes monotonically decreasing loss
- External validation on independent architectures pending

## [0.2.0] — 2026-03-19

- Initial cost predictor (GPU runtime estimation)
- Benchmark calculator
