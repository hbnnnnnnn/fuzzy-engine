# Workspace Organization Manual

## Context

Two workspaces (`ndbao`, `fuzzy-engine`) exist for a VLM project combining Nexus-Gen with an external RAG component for enhanced image editing. Both share the same core components. `ndbao` has ~84 files at root (scripts and SLURM logs mixed together). Goal: consolidate into one canonical workspace with clear categorization, documentation, and an experiment tracking procedure.

---

## Target Structure

Adopt `ndbao` as the single canonical workspace. Merge everything from `fuzzy-engine`, then delete it.

```
ndbao/
├── src/                         # All source code (git repos)
│   ├── nexus-gen/               # ← Nexus-Gen/
│   ├── rag-patch/               # ← rag_patch_training/
│   └── mmdetection/             # ← mmdetection/
│
├── scripts/                     # Shell scripts, grouped by purpose
│   ├── setup/                   # setup_nexus_env.sh, setup_benchmarks.sh, setup_imgedit_geneval.sh
│   ├── build/                   # build_mrag_db.sh, build_journeydb_dataset.sh
│   └── eval/                    # test_retrieval.sh, test_mrag_controlled.sh,
│                                #   test_rag_patch_controlled.sh, run_nexus_test.sh
│
├── benchmarks/                  # Evaluation frameworks (currently under ndbao_hbngoc/)
│   ├── drawbench/
│   ├── geneval/
│   ├── ImgEdit/
│   ├── Janus/
│   ├── Show-o/
│   ├── T2I-CompBench/
│   └── tifa/
│
├── data/                        # Datasets (large files, keep out of git)
│   └── journeydb/               # ← journeydb_dataset/
│
├── experiments/                 # One folder per experiment run
│   └── YYYY-MM-DD_<short-name>/
│       ├── config/              # Copy of the .sh script used to launch this run
│       ├── results/             # ← rag_patch_results_controlled/ output, workdirs/ output
│       └── logs/                # .out / .err SLURM files for this run
│
├── logs/                        # Archive: existing root-level SLURM logs go here
│
├── docs/
│   ├── README.md                # Project overview (see template below)
│   └── EXPERIMENTS.md           # Experiment tracking table (see template below)
│
├── build_journeydb_dataset.py   # Standalone Python builder (keep at root)
└── requirements.txt
```

---

## Migration Steps (in order)

### 1. Merge `fuzzy-engine` → `ndbao`
- For each file/folder that exists in both: keep the one with the later `mtime` (or the one you know is more current).
- Files only in `fuzzy-engine` (e.g., `rag_patch_training/train_rag_patch.sh`): copy them over.
- After verifying nothing was lost, delete `fuzzy-engine/`.

### 2. Create the Folder Skeleton
Inside `ndbao/`, create:
```
src/
scripts/setup/
scripts/build/
scripts/eval/
benchmarks/
data/journeydb/
experiments/
logs/
docs/
```

### 3. Move Source Repos
| From | To |
|---|---|
| `Nexus-Gen/` | `src/nexus-gen/` |
| `rag_patch_training/` | `src/rag-patch/` |
| `mmdetection/` | `src/mmdetection/` |

### 4. Move Shell Scripts
| From | To |
|---|---|
| `setup_nexus_env.sh`, `setup_benchmarks.sh`, `setup_imgedit_geneval.sh` | `scripts/setup/` |
| `build_mrag_db.sh`, `build_journeydb_dataset.sh` | `scripts/build/` |
| `test_retrieval.sh`, `test_mrag_controlled.sh`, `test_rag_patch_controlled.sh`, `run_nexus_test.sh` | `scripts/eval/` |

### 5. Move Benchmarks
Move each subfolder of `ndbao_hbngoc/` directly into `benchmarks/`:
`drawbench/`, `geneval/`, `ImgEdit/`, `Janus/`, `Show-o/`, `T2I-CompBench/`, `tifa/`

Then delete the now-empty `ndbao_hbngoc/`.

### 6. Move Data
`journeydb_dataset/` → `data/journeydb/`

### 7. Reorganize Experiment Results
Each past experiment in `rag_patch_results_controlled/` and `workdirs/` becomes its own folder:
```
experiments/YYYY-MM-DD_<short-name>/
├── config/    ← copy of the .sh script used to launch this run
├── results/   ← output files
└── logs/      ← .out / .err SLURM files
```
Existing root-level `.out`/`.err` SLURM files → move to `logs/` as an archive.

### 8. Write Documentation Files
See templates below.

---

## Documentation Templates

### `docs/README.md`

```markdown
# Nexus-Gen + RAG VLM Workspace

## Goal
Enhance image editing capabilities by combining the Nexus-Gen VLM with an external
multimodal RAG component (MRAG).

## Workspace Layout
| Folder           | Contents                                               |
|------------------|--------------------------------------------------------|
| src/nexus-gen/   | Core model: generation, editing, understanding, MRAG   |
| src/rag-patch/   | Fine-tuning pipeline for RAG patch training            |
| src/mmdetection/ | Object detection framework (used in eval)              |
| scripts/         | Shell scripts: setup/, build/, eval/                   |
| benchmarks/      | Evaluation frameworks (geneval, T2I-CompBench, etc.)   |
| data/            | Raw datasets (not tracked in git)                      |
| experiments/     | One folder per experiment run (config + results + logs)|
| logs/            | Archived SLURM logs not tied to a specific experiment  |

## Key Scripts
- `scripts/setup/setup_nexus_env.sh`  — set up Python environment
- `scripts/build/build_mrag_db.sh`    — build the MRAG retrieval database
- `scripts/eval/run_nexus_test.sh`    — run a full Nexus-Gen evaluation

## Adding a New Experiment
See `docs/EXPERIMENTS.md`.
```

### `docs/EXPERIMENTS.md`

```markdown
# Experiment Log

## Procedure for each new run
1. Create `experiments/YYYY-MM-DD_<short-name>/config/` and copy your `.sh` script there.
2. After the run, move `.out`/`.err` files → `experiments/.../logs/`.
3. Move result outputs → `experiments/.../results/`.
4. Fill in the table below.

| Date | Name | Script | SLURM Job ID | Key Params | Results Path | Status | Notes |
|------|------|--------|--------------|------------|--------------|--------|-------|
```

---

## Naming Conventions Going Forward

| Type | Convention | Example |
|---|---|---|
| Scripts | `<verb>_<target>.sh` | `build_mrag_db.sh`, `eval_geneval.sh` |
| Experiments | `YYYY-MM-DD_<short-description>` | `2026-04-20_rag-patch-lr-sweep` |
| Source repos | lowercase, hyphenated | `nexus-gen/`, `rag-patch/` |
| Data folders | lowercase, hyphenated | `journeydb/`, `mrag-db/` |

---

## Ongoing Tracking Procedure

1. **Before a run**: create `experiments/<date>_<name>/config/` with a copy of your launch script.
2. **After a run**: move SLURM logs → `experiments/.../logs/`, results → `experiments/.../results/`.
3. **Fill in** the `EXPERIMENTS.md` table row.
4. **Tag** significant milestones in git: `git tag -a v0.1 -m "description"`.

Each experiment becomes self-contained and independently reproducible.
