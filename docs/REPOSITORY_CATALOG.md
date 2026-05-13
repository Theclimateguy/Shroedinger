# Repository Catalog and Consolidation Plan

Last audited: 2026-05-13

## 1. Working conclusion

The repository currently mixes three different concerns:

1. Canonical research program code and lightweight reports in `clean_experiments/`
2. Publishable manuscript artifacts at the repository root (`main.pdf`, `main_ru.pdf`)
3. Local manuscript workspace and build byproducts in `manuscript/` and other local-only folders

The scientific body of work is not fragmented across branches. The branch fragmentation is mostly a packaging problem.

## 2. Branch audit

| Branch | Status | Practical meaning | Recommendation |
|---|---|---|---|
| `gen1` | historical branch | earlier documentation-clean snapshot before the later `A11..A15` and release finalization work landed | preserve as historical generation only |
| `gen2` | current `origin/HEAD`, tag lineage `v2.2` | fullest and most up-to-date research corpus; includes canonical `T01..T20`, `A01..A15`, continuation logs, reports, citation files, and root PDFs | current canonical working generation |
| `gen3` | historical branch, tag lineage `v1.0` | trimmed Zenodo-style release branch with many docs/results removed; this is a packaging variant, not a distinct research generation | preserve as historical generation only |
| `gen4` | newer remote snapshot | branch carrying the repository-catalog / branch-restructuring step before the Zenodo README update landed on `gen2` | preserve if needed for branch-history bookkeeping |
| `codex/main_gen2-cleanup-backup` | local helper branch | local cleanup snapshot | archive locally if needed, otherwise not part of final branch model |
| `codex/repack-clean-main` | local helper branch | local repack snapshot | archive locally if needed, otherwise not part of final branch model |

Legacy alias map:

- `main` -> `gen1`
- `main_gen2` -> `gen2`
- `main_gen3` -> `gen3`

Consolidation note:

- the only meaningful code delta found in legacy `main_gen3` / current `gen3` was the optional causal mode-selection parameter `mode_select_end_year` in `clean_experiments/experiment_M_cosmo_flow.py`
- that patch has been merged into the full working line, so `gen3` does not need to be treated as a separate source of scientific logic

## 3. Research-program structure already present

The repository already has a strong internal scientific taxonomy:

- `TOY_MODEL`: `T01..T20`
- `ATMOSPHERE_DATA`: `A01..A15`
- continuation / extension runs:
  - `A05.R*`
  - `A07.R*`
  - `A11.E*`

Primary sources of truth:

- `research_programm_summary.csv`
- `clean_experiments/EXPERIMENT_NUMBERING.md`
- `clean_experiments/HYPOTHESIS_ROADMAP.md`

This means the main task is not to redesign the science tree, but to expose it cleanly at the repository level.

## 4. Artifact layers

### 4.1 Canonical tracked layer

Should remain in the GitHub-facing canonical branch:

- `README.md`
- `LICENSE`
- `CITATION.cff`
- `main.pdf`
- `main_ru.pdf`
- `research_programm_summary.csv`
- `clean_experiments/`
- `docs/`

### 4.2 Local manuscript workspace

Currently local-only and already ignored by Git:

- `manuscript/main.tex`
- `manuscript/main_eng.tex`
- `manuscript/references*.bib`
- `manuscript/figures/*`

Important duplication:

- `main.pdf == manuscript/main_eng.pdf`
- `main_ru.pdf == manuscript/main.pdf`

Interpretation:

- root PDFs are the tracked publication artifacts
- `manuscript/` is the local editable source workspace

### 4.3 Local generated/runtime artifacts

These are not part of the scientific source tree and should stay outside the canonical tracked layer:

- `clean_experiments/out/`
- `clean_experiments/__pycache__/`
- TeX build byproducts from `manuscript/`
- future temporary parking under `legacy/`

## 5. Target repository model

Recommended steady-state layout:

```text
Shroedinger/
├── README.md
├── LICENSE
├── CITATION.cff
├── main.pdf
├── main_ru.pdf
├── research_programm_summary.csv
├── docs/
│   └── REPOSITORY_CATALOG.md
├── clean_experiments/
│   ├── canonical scripts
│   ├── experiment manifests
│   └── markdown reports
├── manuscript/          # local-only source workspace, gitignored
└── legacy/              # local-only parking for noncanonical local artifacts, gitignored
```

## 6. Branch policy after consolidation

Recommended branch model:

1. use generational names `genN` for long-lived public branches
2. current canonical working branch is `gen2`
3. preserve release tags such as `v1.0`, `v2.0`, `v2.1`, `v2.2`, `v3.0` as archival markers
4. if a new long-lived branch is needed, name it `gen(N+1)` relative to the highest existing generation
5. if a compact Zenodo release is needed again, prefer tags/releases over creating a separate packaging-only branch unless the branch itself is part of the archival logic

## 7. Immediate local cleanup policy

Safe to move into `legacy/` immediately:

- runtime caches
- TeX build byproducts
- other non-source local artifacts

Not to move yet without explicit scientific review:

- any canonical `T*` or `A*` script
- continuation scripts `A05.R*`, `A07.R*`, `A11.E*`
- markdown reports under `clean_experiments/results/`
- root publication PDFs

## 8. Next actions

1. keep `gen2` as the local working branch unless a new generation is intentionally created
2. keep local/remote naming synchronized with the public `genN` scheme
3. create the next long-lived branch only as `gen(N+1)`
4. keep manuscript, runtime caches, and temporary parking in local-only ignored paths
5. treat `gen1`, `gen3`, and `gen4` as historical reference branches unless there is an explicit need to reopen them

## 9. Local alignment status

Current local Git alignment:

- local current branch renamed from `main_gen2` to `gen2`
- local historical branches renamed from `main` / `main_gen3` to `gen1` / `gen3`
- local `gen2` tracks `origin/gen2`
- local `gen1` tracks `origin/gen1`
- local `gen3` tracks `origin/gen3`
- local `gen4` tracks `origin/gen4`
- `origin/HEAD` now points to `origin/gen2`

Reference commands used for local alignment:

```bash
git fetch origin --prune
git branch -m main_gen2 gen2
git branch -m main gen1
git branch -m main_gen3 gen3
git branch --set-upstream-to=origin/gen2 gen2
git branch --set-upstream-to=origin/gen1 gen1
git branch --set-upstream-to=origin/gen3 gen3
git branch --track gen4 origin/gen4
git remote set-head origin -a
```

## 10. Repository size note

Current local size is dominated by Git history, not the working tree:

- working directory content is only a few megabytes
- `.git` is about `3.2G`
- `git count-objects -vH` reports one packed history around `3.16 GiB`

Implication:

- deleting local branches will simplify navigation
- deleting branches alone will not materially shrink `.git`

If real size reduction is required after branch consolidation, use one of:

1. fresh clean clone after the branch model is stabilized
2. deliberate history rewrite with `git filter-repo` or BFG, only if you want to rewrite GitHub history
