# Background-paper validation

Reviewed 28 September 2026 after the core refactor and its documentation were
completed and pushed through `900d63b`.

- 32 PDF pages, including cover and references; approximately 10,000 source words.
- 39 primary-source bibliography entries, all cited; no missing citation keys or
  duplicate explicit labels.
- Five attributed reproductions from major papers, four original synthesis
  diagrams, and four analytical plots. Page layouts and reproduced figures
  were rendered and visually inspected. Source hashes and licenses are recorded
  in `figures/papers/manifest.json`.
- Native LaTeX equations, vector figures, and original raster artwork; the final build has no overflowing
  boxes, unresolved references, missing glyphs, or duplicate-label warnings.
- Worked KV-byte examples checked independently: 8192 tokens = 1 GiB and
  8704 tokens = 1.0625 GiB under the stated 128 KiB/token geometry.
- Source bundle rebuilt independently using its included vector figures.
- Documentation-only work after the refactor's full suite: 420 passed, five
  skipped, six warnings. No additional serving-performance or GPU result is
  implied by building this paper.

Research scope is deliberately bounded: primary-source mechanisms and relevant
current systems, not an exhaustive catalogue or a cross-paper speed ranking.
Living documentation is dated, not release-pinned. Recent preprints are identified
as research rather than production guarantees. Numerical plots are illustrative;
repository results are linked separately and retain their limitations.

## Measured-evidence addendum, 30 September 2026

- Added Section 6.6 (Table 3, Figure 14) from four dated GPU reports of 29-30
  September; numbers and caveats were checked against those reports. The
  literature survey and its cutoff are unchanged.
- Figure 14 is the only measured plot; its data are in `measured-evidence.json`.
- 36 PDF pages. The build has no overflowing boxes or unresolved references;
  the changed pages (title, Section 6.6, Appendix E) were rendered and inspected.
- No ATFM policy speedup is claimed; stage E is under evaluation.
