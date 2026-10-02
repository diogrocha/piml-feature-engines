#!/usr/bin/env bash
# Downloads the four C-MAPSS training files into data/CMAPSSData if missing.
# Official source: NASA Prognostics Center of Excellence data repository.
# This script uses a public GitHub mirror of the same files; if you already have the
# official files, place train_FD00{1..4}.txt in data/CMAPSSData and skip it.
set -e
cd "$(dirname "$0")"
mkdir -p data/CMAPSSData
for f in train_FD001.txt train_FD002.txt train_FD003.txt train_FD004.txt; do
  [ -s "data/CMAPSSData/$f" ] || curl -sfL -o "data/CMAPSSData/$f" \
    "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/$f"
done
wc -l data/CMAPSSData/train_FD00*.txt   # expected 20631, 53759, 24720, 61249
