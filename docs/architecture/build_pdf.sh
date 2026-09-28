#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tmp/pdfs/architecture output/pdf
for source in docs/architecture/figures/*.svg; do
  inkscape "$source" --export-text-to-path \
    --export-filename="tmp/pdfs/architecture/$(basename "${source%.svg}").pdf" 2>/dev/null
done
pandoc docs/architecture/03-code-and-documentation.md \
  --from=markdown+gfm_auto_identifiers --standalone --toc --toc-depth=2 --pdf-engine=xelatex \
  --lua-filter=docs/architecture/pdf-layout.lua \
  --include-in-header=docs/architecture/pdf-layout.tex \
  --metadata title='ATFM: code and documentation architecture' \
  --metadata date='Implementation baseline: 28 September 2026' \
  -V documentclass=article -V papersize=letter -V fontsize=10pt \
  -V geometry:margin=0.6in -V mainfont='DejaVu Serif' \
  -V sansfont='DejaVu Sans' -V monofont='DejaVu Sans Mono' \
  -V monofontoptions=Scale=0.8 -V colorlinks=true \
  -o output/pdf/atfm-architecture.pdf
cp output/pdf/atfm-architecture.pdf docs/architecture/atfm-architecture.pdf
