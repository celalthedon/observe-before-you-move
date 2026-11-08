# OBYM V3 methodology and reproduction

## Frozen experiment

The V3 release uses protocol `94fd47e48b4f` (`v3_first_eligible_final_v1`). The freeze ID is the first 12 hex digits of the SHA-256 of the protocol serialized as sorted JSON with the original Python settings. The archived `protocol.json` bytes have SHA-256 `2042f839866917fa3292c225049877b3f4d4ffd3a0155d936a5a543844b42d00`.

| Split | Source groups | Variants | Seeds |
| --- | ---: | ---: | --- |
| Training | 24 | 96 | 1000–1023 |
| Validation | 6 | 24 | 2000–2005 |
| Test | 20 | 80 | 4000–4019 |

Each source contributes both mirror settings crossed with both travel directions. The test covers seven barrier, seven gate-open, and six gate-closed source layouts from the same three synthetic templates used in the other splits. The archived preflight checked seed and geometry separation from the historical experiment inventory and saved validation model checks before collection. The evidence is scoped to the simulated controller, candidate actions, camera and costs in this protocol.

## One decision and three action branches

The NoView episode records decision states. Eligibility requires a path of at least two cells and a route-relevant unknown region. At the first eligible state, the policy selects one of three actions, 0°, −45°, or +45°. A 0° choice commits to navigation with routine depth sensing. For a turn, the controller takes one optional observation, updates occupancy and replans. Both fixed-turn continuations verify the same pre-decision prefix against NoView. All 80 archived test variants were first eligible at decision index 0. The first-eligible rule applies even when eligibility is later in an episode.

The two candidate encodings each contain seven binary 80×80 channels and six pose values. Learned policies predict signed net seconds saved for each turn. They turn only when the maximum exceeds the frozen threshold, with ties between candidates resolved toward −45°. The three fixed-action baselines execute the branches. HeadingRule uses −45° for initial headings facing west and 0° otherwise. The other five policies select one of the stored branch outcomes for each variant. This gives 240 executed test episodes and 720 policy outcomes, of which 480 reuse a branch for another policy. Three separate train-parity episodes are outside these test counts. The online learned-policy route in the historical V2 evaluator is a separate workflow.

## Training and model selection

The target for an optional view is the paired NoView time minus single-view time from the same first eligible state. The dataset directory holds 120 per-variant NPZs. The frozen `dataset_manifest.json` is the inherited scene/variant manifest and describes the older V2 three-state sampler. The V3 `collection_protocol.json` and recovered collection cell specify the first-eligible sampling used for these tensors.

The three ridge families use pose only, whole-patch channel means plus pose, or pooled spatial channel means plus pose. The executed spatial job shuffled the 24 training sources with seed 20261108 into six disjoint folds of four sources. It selected configuration by out-of-fold decision regret, followed by harmful looks, level, descending alpha and descending threshold. SpatialRidgeCV selected the 1×1, 2×2 and 4×4 pooling grids, alpha 100, and a 0 s threshold. The 24 validation variants supplied a separate parity reference. The final test supplied no tuning inputs.

The saved CNN training run used three seeds (11, 23, 47) for each of two ensembles, with 60 epochs per member and training target scale 7.443235397338867 s. One ensemble used mean squared error and the other also used a ranking loss weighted at 0.5. Both used a 0.15 s decision threshold. The training checkpoint has SHA-256 `b8d637bd5be7c725454a26e9354cb7e4c1967d879126221ca15e0d258737b0d1`. The bundled `cnn_models.pt` is byte-identical to the saved run's `models.pt`. The original CNN cell (zero-based index 79, ID `3db1b240` in `01_day1_simulation.ipynb`) and dataset collection cells are preserved with [notebook and execution provenance](../tools/v3/provenance/notebook_evidence.json).

## Outcomes and uncertainty

A safe success reaches `GOAL` with zero collision attempts. Any other outcome is assigned 180 s for the primary mean. All 240 test branches were safe successes. Simulated time accounts for navigation, turning and sensing. Warm CPU inference wall time is a separate diagnostic and is excluded from the primary metric. Regret is the chosen branch cost minus the lowest of the three archived costs. This minimum is a retrospective reference. A look counts as harmful when its cost exceeds the same variant's NoView cost by more than `1e-6` s.

For each baseline, subtract the SpatialRidgeCV cost from the baseline cost within each variant, then average the four paired differences within each source. Resample 20 sources with replacement for each of 10,000 bootstrap draws using NumPy's `default_rng(20261108)`. The reporter reuses the same resample indices across comparisons and takes the 2.5th and 97.5th percentiles. Positive differences favor SpatialRidgeCV. The eight 95% intervals are unadjusted for multiple comparisons.

| Baseline vs SpatialRidgeCV | Mean time saved (s) | 95% interval (s) |
| --- | ---: | ---: |
| NoView | +1.381 | [0.402, 2.515] |
| FixedMinus45 | +1.122 | [0.420, 1.969] |
| FixedPlus45 | +0.494 | [−0.435, 1.585] |
| HeadingRule | +1.419 | [0.437, 2.558] |
| PoseRidgeCV | +0.597 | [0.063, 1.303] |
| MeanMapRidgeCV | +0.597 | [0.063, 1.303] |
| CNN_MSE | +0.348 | [−0.354, 1.274] |
| CNN_MSE_Rank | +0.307 | [−0.436, 1.245] |

SpatialRidgeCV had the lowest observed mean. The intervals for FixedPlus45, CNN_MSE and CNN_MSE_Rank include zero.

## Evidence and integrity

- [Frozen protocol, checkpoints, and validation records](../data/v3/results/v3-final-protocol/94fd47e48b4f) include the preflight test manifest, model hashes and train-selection record.
- [Final case records](../data/v3/results/v3-final-test/94fd47e48b4f/cases) preserve branch summaries and hashes, NoView snapshots, and encoded model inputs. [Original reports](../data/v3/results/v3-final-test/94fd47e48b4f/reports) include all policy rows, predictions, and a report completion marker.
- [Training and validation tensors](../data/v3/results/v3-initial-dataset) are the original NPZ files. [Training-run evidence](../data/v3/results/v3-training/20261107_014522_557567) records the CNN configuration and validation outputs.
- [Pinned simulator source](../data/v3/pinned) preserves the nine source files byte-for-byte, including their original line endings. The collector applies the archived initial-centering AST transformation to the PostTurnReplan runner and checks the resulting controller hash `891d968c0400e787a4add611bc8903c92497a70f54ba2ac21341419c7d20be65`.
- The [release manifest](../data/v3/release_manifest.json) indexes every bundled core file and every member of both detached trace ZIPs. The original collector and reporter hashes link the saved job scripts to the completion records. The saved spatial and preflight workers and their run logs are retained under `data/v3/results`.

The detached trace ZIPs contain original simulator outputs. Keep both files together to check every member and the original final episode hashes directly, without extracting:

```powershell
python tools/verify_v3_release.py --trace-archives path/to/release-assets
```

| Release asset | Contents |
| --- | --- |
| `obym-v3-94fd47e48b4f-test-traces.zip` | 1,944 files from 240 final test and three train-parity episodes |
| `obym-v3-94fd47e48b4f-dataset-traces.zip` | 2,880 files from 360 train and validation collection episodes |

Extract either ZIP at the artifact root (`data/v3` by default) to restore its `results/…/results/day3` tree. Original JSON retains historical machine paths for provenance. The portable checks locate archived records by relative layout and run ID. Use `--artifact-root` with the verifier when the bundle is stored elsewhere. Treat the Git release and the frozen hashes as the trust roots when obtaining detached assets.

## Reproduction commands

Run these commands from the repository root with Python 3.12. Each generated report uses a fresh output directory. Pass `--artifact-root` to point to a relocated bundle and `--source-root` to override the pinned source directory with the same verified bytes.

### Archived statistics

```powershell
python -m pip install -r requirements-v3-results.txt
python tools/verify_v3_release.py
python -m tools.v3 stats --output results/generated/v3/stats
```

This workflow uses NumPy, the compact branch records and archived prediction scores. It reconstructs the choices, all 720 outcomes, 180 source-policy rows, the policy summary, and all eight bootstrap intervals. It checks those values against the original report. Generated tables use portable episode references and omit historical machine-dependent timing. The original timing fields, figures and scores remain in the archived report. After extracting the test trace ZIP, add `--deep` to hash the underlying simulator outputs as well.

### Frozen model predictions and preflight

```powershell
python -m pip install -r requirements-v3-model.txt
python -m tools.v3 predictions --output results/generated/v3/predictions
python -m tools.v3 preflight --output results/generated/v3/preflight
```

The prediction command runs the three frozen ridge models and two CNN ensembles on CPU. It compares all 24 validation and 80 held-out actions per model, validation ridge probes, and both held-out candidate scores. Score tolerances are 1e-6 s for ridge and 1e-4 s for CNN. Preflight reconstructs the 80 held-out cases from pinned geometry code, checks separation from the historical manifests, and repeats the validation probes. It accepts additional manifest directories via `--historical-root` and can also compare validation predictions on CPU and CUDA with `--cuda`. Both commands use saved inputs and need no simulator.

### Optional branch collection and retraining

These workflows explicitly opt in to simulation or fitting. On Windows, the archived simulator uses the conda-forge PyBullet 3.25 package family. Activate the environment so the Python process can locate its runtime DLLs. The local OBYM environment used for release checks has Python 3.12.15 and PyBullet package build `py312h3d6c809_5` for Windows. The archived report records CPU inference with PyTorch `2.11.0+cu128`.

```powershell
conda env create -f environment-simulation.yml
conda activate obym-simulation
python -m tools.v3 collect-final --execute --output results/generated/v3/new-final --batch-limit 1 --minutes 5
python -m tools.v3 collect-dataset --execute --output results/generated/v3/new-dataset --batch-limit 1
```

The final collector checks the three train-parity branches before collecting the next held-out batch. Reuse the same output directory to resume. The time budget stops the collector from starting another episode after the deadline. The original dataset collector uses only training and validation scenes. A platform with compatible PyBullet binaries or a build toolchain can install `requirements-v3-simulation.txt` instead of the Conda environment.

```powershell
python -m pip install -r requirements-v3-model.txt
python -m tools.v3 train-spatial --execute --output results/generated/v3/new-ridge
python -m tools.v3 train-cnn --execute --output results/generated/v3/new-cnn --device cuda
```

Retraining defaults to the archived NPZ tensors and writes new checkpoints. The CNN run used CUDA. Refit weights and regenerated simulator logs can differ with package builds, hardware, and timing. The frozen checkpoint hashes identify the released weights. The saved training cell contains completion output and a generic notebook kernel-crash marker. Its output checkpoint and validation results are available. A complete lockfile and hardware image for the original CUDA training run were unavailable. The saved dataset collector logs jump from 12 to 15 existing cases between cells. The intervening invocation record is unavailable. All 120 tensors and 360 collection episode directories are preserved.

## Release verification

- Core integrity passed on a clean Git export. The verifier checked 734 files, the original protocol, preflight and report links, the frozen controller AST and training hashes, 80 test-case completion markers, and 243 distinct test and train-parity branch records. Both trace ZIPs passed checksum and member verification (4,824 files).
- Archived statistics reproduced all 720 policy outcomes, 180 source-policy rows, and eight unadjusted bootstrap intervals. The independent audit checked every selected branch and reproduced the intervals to within 1e-12 s.
- CPU parity matched 24 validation and 80 held-out actions per model. With PyTorch `2.11.0+cu128` and NumPy 2.5.4, all five models matched the held-out scores exactly. A second CPU build (`2.6.0+cu124`, NumPy 2.0.2) had maximum held-out score errors of 1.78e-15 s for ridge and 2.74e-6 s for CNN, with identical choices.
- Preflight matched all 80 held-out cases, checked the 50 historical seeds, and executed zero navigation episodes. Independent audit also reconstructed the 80 first-eligible encodings from the saved NoView snapshots and matched the model inputs exactly.
- Six V3 unit tests passed with `python -B -m unittest discover -s tools/v3/tests -v`. The collector setup validated the pinned simulator configuration and paused before starting an episode.

Release verification used archived inputs and CPU inference. Fresh branch collection and retraining are optional workflows.
