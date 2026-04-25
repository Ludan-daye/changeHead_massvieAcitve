# NeurIPS 2026 — Function Words as Geometric Anchors

A polynomial SVD mechanism of massive activations across 26 transformer language models.

## Directory layout

```
paper/
├── main.tex              # NeurIPS 2026 entry point; contains Abstract + Sections 1-7 inline
├── appendix.tex          # Appendices (no page limit)
├── references.bib        # BibTeX bibliography
├── neurips_2026.sty      # Official style file (do NOT modify)
├── checklist.tex         # Official NeurIPS checklist (must be included; desk-reject if removed)
├── figures/              # Figures (drop PDFs / PNGs here)
└── README.md             # This file
```

Only three content files need your attention: `main.tex`, `appendix.tex`, `references.bib`.
The `.sty` and `checklist.tex` are required template infrastructure; they must ship with
the submission but you should not edit them.

## Build

```powershell
# latexmk (recommended)
latexmk -pdf -interaction=nonstopmode main.tex

# or the explicit sequence
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

`pdflatex` is not in the current `PATH` on this machine. Install
MiKTeX (<https://miktex.org>) or TeX Live and re-run.

The `[preprint]` option inside `main.tex` keeps author identity visible while
drafting. **Remove it before submitting** so that the submission is
anonymised and carries line numbers (required by NeurIPS 2026).

## Drafting status

| Section                       | Status     |
|-------------------------------|------------|
| Abstract                      | drafted    |
| 1 Introduction                | drafted    |
| 2 Related Work                | skeleton   |
| 3 Problem Formulation & Setup | skeleton   |
| 4 Method                      | skeleton   |
| 5 Experiments (RQ1--RQ6)      | skeleton   |
| 6 Discussion                  | skeleton   |
| 7 Conclusion                  | skeleton   |
| Appendix A--F                 | skeleton   |
| References                    | 33 entries |

All remaining writing tasks are marked inline with `\todonote{...}`. Use
`grep` / `rg` to locate them:

```powershell
Select-String -Path 'main.tex','appendix.tex' -Pattern 'todonote'
```

## Source-material pointers (26-model experiments)

The paper is built around the **26-model final report**, not the 8-model
ACL version. Canonical locations (repo root = `..`):

| Item                                   | Path |
|----------------------------------------|------|
| Aggregated master summary              | `../changeHead_massvieAcitve/final_report/aggregated/SUMMARY_26_MODELS.md` |
| Machine-readable master JSON           | `../changeHead_massvieAcitve/github_submission/aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json` |
| RQ1 (attention ablation)               | `../changeHead_massvieAcitve/final_report/RQ1_attention_ablation/ANALYSIS.md` |
| RQ2 (MLP source, retain-ratio)         | `../changeHead_massvieAcitve/final_report/RQ2_mlp_source/ANALYSIS.md` |
| RQ4 (SVD polynomial fit)               | `../changeHead_massvieAcitve/final_report/RQ4_svd_alignment/ANALYSIS.md` |
| RQ5 (V-matrix ablation)                | `../changeHead_massvieAcitve/final_report/RQ5_v_ablation/ANALYSIS.md` |
| ACL camera-ready text (reviewer fixes) | `../changeHead_massvieAcitve/paper/acl_source/Massive activation.tex` |
| Per-doc robustness (C4 N=64)           | `../changeHead_massvieAcitve/results_per_doc/` |
| Chinese modification logs              | `../changeHead_massvieAcitve/paper/notes_zh/` |

The old 8-model IEEE-format paper lives at the repo root as
`../Massive activation old.tex` and is kept only for reference while porting
reviewer-addressed language.

## Anonymisation (before submission)

1. In `main.tex` switch the style option:
   ```latex
   \usepackage{neurips_2026}          % remove "preprint"
   ```
2. Delete the author block / acknowledgements.
3. Ensure `figures/` contains no identifying metadata.
4. Ensure `\href` / `\url` do not link to identifying repositories.
5. Fill every `\answerTODO{}` / `\justificationTODO{}` in `checklist.tex`.
