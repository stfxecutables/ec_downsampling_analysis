#!/bin/bash
IN=methods.md
BIB=methods.bib
OUT=methods.html
STYLE=mdstyle.html
CITESTYLE=apa.csl

# --citeproc: process citations in --bibliography
# --mathjax: include mathjax js
# --standalone: more compatible with mathjax than --self-contained
# --resource-path: for e.g. included images, paths are relative to here
# --metadata: additional pandoc args
# --metadata: link-citations: link citations in text to bottom refs
# --metadata: link-bibliography: turn DOI / links to working hrefs
# --metadata: pagetitle: for HTML title output
# -H: include verbatim in html header
# -s: make output a full document and not a document fragment
# https://pandoc.org/MANUAL.html#extension-citations for more info
# bib file should be BibLaTeX
# Note: make sure you have latest pandoc (e.g. 2.18+)

pandoc \
    --standalone \
    --mathjax \
    --resource-path . \
    --citeproc \
    --bibliography $BIB \
    --csl $CITESTYLE \
    --metadata link-citations=true \
    --metadata link-bibliography=true \
    --metadata lang=en-US \
    -f markdown+citations -t html $IN \
    --metadata pagetitle="Methods" \
    -H $STYLE \
    -s -o $OUT
echo -n "Built at $(date): " &&\
if ! command -v readlink &> /dev/null;
then
    greadlink -f $OUT;
else
    readlink -f $OUT;
fi
