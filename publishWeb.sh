#!/bin/bash
set -euo pipefail

mkdir -p public
cp -ru figures pdfs style codes dae2021 .nojekyll public/ 2>/dev/null || true

pandoc index.org \
  -f org -t html5 --citeproc \
  --csl=style/APA-CV.csl \
  -M suppress-bibliography=true \
  --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js \
  --css=style/myorg.css \
  -s -o public/index.html

sed -i 's/Wang, H\./<strong>Wang, H.<\/strong>/g' public/index.html
