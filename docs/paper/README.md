# LKOS Whitepaper v0.10.0 — LaTeX Source

This directory contains the complete LaTeX source for
`LKOS_Whitepaper_v0.10.0.pdf` (13 pages: cover + TOC + 11 content pages,
2 TikZ figures, 3 booktabs tables, 11 references). The compiled PDF ships
separately; this tree is the reproducible source of record.

## Layout

| File | Role |
|---|---|
| `main.tex` | Master document: `article` class, 11pt, TOC-first (no `\maketitle`), includes the four section modules |
| `sec1-intro.tex` | §1 Introduction: motivation, local-first thesis, contributions |
| `sec2-arch.tex` | §2 Architecture: storage core, hybrid retrieval, knowledge layer, pipeline diagram (TikZ) |
| `sec3-results.tex` | §3 Evaluation: correctness gates, ANN crossover results, latency profile |
| `sec4-close.tex` | §4 Limitations & closing + `thebibliography` (11 entries) |
| `cover.html` | Standalone cover page (794×1123 px, palette `#121211` / `#cbac4d`), rendered to PDF separately and prepended |

## Building

Requires [Tectonic](https://tectonic-typesetting.github.io/) (or `latexmk`).
Run the compiler **twice** so the table of contents and cross-references
converge:

```sh
tectonic main.tex && tectonic main.tex
# or: latexmk -pdf main.tex
```

Output: `main.pdf` (12 pages, TOC + body).

## Cover merge

The cover is HTML, not LaTeX. Render `cover.html` at exactly 794×1123 px,
scale the result to A4, and **prepend** it to the body:

```python
from pypdf import PdfReader, PdfWriter, Transformation

body = PdfReader("main.pdf")
cover = PdfReader("cover.pdf")
w = PdfWriter()
page = cover.pages[0]
page.add_transformation(Transformation().scale(
    595.276 / float(page.mediabox.width),
    841.89 / float(page.mediabox.height)))
w.add_page(page)          # prepend cover
for p in body.pages:
    w.append(p)           # append() preserves TOC/citation link annotations
```

**Warning:** do not use `add_page()` for the body pages — it drops all
hyperlink annotations (TOC and reference links break). Use `append()`
for every page after the cover.

## Provenance contract

Every number in `sec3-results.tex` comes from `benchmarks/results/*` in the
repository root; test counts follow `docs/TESTING.md`; the version string in
`main.tex` matches the release tag. If documentation and code drift apart,
that is a release blocker — fix the source here and rebuild, never hand-edit
the PDF.
