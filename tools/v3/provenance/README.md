# Recovered V3 execution source

The preserved notebook cells come from `observe-before-you-move/01_day1_simulation.ipynb`.
`notebook_evidence.json` records the full notebook hash, exact cell source hashes,
embedded worker hashes, and saved output. The Python files preserve the cell
source exactly, including the original launch paths.

- Cells 76 and 77 embed identical dataset collectors. Cell 76 logs completion
  of the first 12 cases. Cell 77 starts with 15 completed cases and logs the
  final 96 train and 24 validation cases. The saved invocation record for the
  three intervening cases was not located. The timestamped archive identifies
  them as `train_layout_1003_myP_LR`, `train_layout_1003_myP_RL`, and
  `train_layout_1003_myM_LR`.
- Cell 79 records all six 60-epoch fits on CUDA and the successful worker
  completion marker for `20261107_014522_557567`. The notebook also retains a
  generic kernel crash marker for the current or previous cell. The archived
  training `models.pt` and frozen `cnn_models.pt` have identical SHA-256
  `b8d637bd5be7c725454a26e9354cb7e4c1967d879126221ca15e0d258737b0d1`.

The release also preserves original worker files and logs under
`data/v3/results`.

| Run | Source relative to `data/v3/results` | Evidence |
| --- | --- | --- |
| Spatial ridge | `v3-spatial-jobs/20261108_190835_929689/worker.py` | Six-fold, source-grouped training selection and validation output for run `20261108_190836_472908` |
| Initial preflight | `v3-preflight-jobs/20261108_200245_290060/worker.py` | CNN CPU/CUDA difference exceeded tolerance |
| IEEE FP32 preflight | `v3-preflight-jobs/20261108_201819_531119/worker_ieee.py` | Successful preflight for the frozen manifest |
| Final collection | `v3-final-jobs/{20261108_204438_019875,20261108_215534_579066}/collector.py` | Identical scripts. The first batch reached 51/80 cases, and the resumed batch reached 80/80 |
| Final report | `v3-report-jobs/20261108_232050_711219/reporter.py` | Compact and full-trace integrity checks, frozen predictions, and published statistics |

The review ZIP and neighboring notebook did not contain a standalone freeze
assembly worker. The release preserves the resulting frozen protocol and its
artifact and source hashes. Portable commands consume that fixed protocol.
