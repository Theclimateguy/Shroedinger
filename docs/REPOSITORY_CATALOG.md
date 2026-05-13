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
| `main_gen2` | current `origin/HEAD`, tag `v2.2` | fullest and most up-to-date research corpus; includes canonical `T01..T20`, `A01..A15`, continuation logs, reports, citation files, and root PDFs | use as the canonical base for consolidation |
| `main` | older tracked branch | transitional documentation-clean snapshot before the later `A11..A15` and release finalization work landed | fold into canonical `main`; no need to keep as a separate scientific line |
| `main_gen3` | trimmed release branch, tag `v1.0` | minimal Zenodo-style code release with many docs/results removed; this is a packaging variant, not a distinct research generation | preserve via tag/history only; not needed as a long-lived branch |
| `codex/main_gen2-cleanup-backup` | local helper branch | local cleanup snapshot | archive locally if needed, otherwise not part of final branch model |
| `codex/repack-clean-main` | local helper branch | local repack snapshot | archive locally if needed, otherwise not part of final branch model |

Consolidation note:

- the only meaningful code delta found in `main_gen3` was the optional causal mode-selection parameter `mode_select_end_year` in `clean_experiments/experiment_M_cosmo_flow.py`
- that patch has been merged into the current working tree, so `main_gen3` no longer appears to contain unique scientific logic

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

1. `main` = full canonical research branch
2. keep `v1.0`, `v2.0`, `v2.1`, `v2.2` as historical release markers
3. remove long-lived packaging branches after `main` is aligned and remote references are updated
4. if a compact Zenodo release is needed again, create it from tags/releases, not as a persistent parallel branch

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

1. Keep `main_gen2` content as the effective canonical baseline
2. Add `legacy/` to `.gitignore`
3. Move obvious local-only artifacts into `legacy/`
4. Update root navigation so the branch policy and local-vs-tracked split are explicit
5. Prepare a safe GitHub consolidation path:
   - fast-forward or replace `main` with the `main_gen2` content
   - repoint default branch
   - only then retire `main_gen2` and `main_gen3`

## 9. Safe Git consolidation sequence

Recommended order:

1. commit the current cleanup on top of the `main_gen2` line
2. verify that `main_gen3` contributes no remaining unique scientific logic beyond the already merged `mode_select_end_year` patch
3. update `main` so it points to the full canonical commit
4. switch GitHub default branch from `main_gen2` to `main`
5. only after validation, retire `main_gen2` and `main_gen3` as long-lived branches while keeping release tags

Suggested command sequence after review:

```bash
git checkout main_gen2
git checkout -b codex/repo-consolidation-main
git add .gitignore README.md docs/REPOSITORY_CATALOG.md clean_experiments/experiment_M_cosmo_flow.py
git commit -m "Catalog repository and prepare mainline consolidation"

git branch -f main HEAD
git checkout main
git push origin main
```

Then on GitHub:

1. set `main` as the default branch
2. confirm CI / release assets / Zenodo expectations
3. optionally delete remote `main_gen2` and `main_gen3`

Local cleanup after remote confirmation:

```bash
git branch -d main_gen3
git branch -d main_gen2
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
