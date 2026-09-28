# Background-paper validation

Reviewed 28 September 2026 after the core refactor and its documentation were
completed and pushed through `900d63b`.

- 30 PDF pages, including cover and references; approximately 10,000 source words.
- 39 primary-source bibliography entries, all cited; no missing citation keys or
  duplicate explicit labels.
- Six original architecture diagrams and four analytical plots. All final pages
  were rendered and visually inspected; crowded labels found in the initial
  agent/disaggregation diagrams were corrected and rechecked.
- Native LaTeX equations and vector figures; the final build has no overflowing
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
