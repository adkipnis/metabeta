#!/bin/bash
# Usage: ./scripts/generate-family.sh <family>   (n=normal, b=bernoulli, p=poisson)

FAMILY_NAME="${1:-}"
[[ -z "$FAMILY_NAME" ]] && { echo "Usage: $0 <family>  (n, b, or p)"; exit 1; }

case "$FAMILY_NAME" in
    n) FAMILY=0 ;;
    b) FAMILY=1 ;;
    p) FAMILY=2 ;;
    *) echo "Unknown family '$FAMILY_NAME' — use n, b, or p"; exit 1 ;;
esac

for SIZE in small medium large huge; do
    python generate.py --size "$SIZE" --family "$FAMILY" --ds_type mixed --partition train --epochs 2 --loop
    python generate.py --size "$SIZE" --family "$FAMILY" --ds_type sampled --partition eval --bs_valid 8192 --bs_test 8192 --loop
done
