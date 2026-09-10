#!/usr/bin/env bash
#
# Reproduce the Levy false-positive benchmark end to end.
#
# Every number in the report comes from this sequence. It is one GPU pass per
# condition; all the parameter searches (beta, k, drop thresholds, AUSE) replay
# offline on CPU from the per-box dumps, which is why there are six passes and
# not sixty.
#
#   ./opencood/tools/reproduce_levy_benchmark.sh [MODEL_DIR]
#
# Default MODEL_DIR is the stage-1 logit base. Expect ~2.5 h on one 4090:
# ~20 min per pass, seconds for the offline analysis.
#
# Requires: the model_dir's config.yaml must carry the `levy.uncertainty_variants`
# block (see hypes_yaml/.../m1_pyramid_levy_logit.yaml) or the cosine channels
# will be absent and the follow-up sections will be empty.
set -euo pipefail

MODEL_DIR="${1:-opencood/logs/Pyramid_m1_base_levy_logit}"
FUSION=intermediate
RANGE="102.4,102.4"
OUT="$MODEL_DIR/levy_benchmark"
INFER="python opencood/tools/inference.py --model_dir $MODEL_DIR --fusion_method $FUSION"

# Epoch suffix inference.py appends to every artefact path. Read it back from
# whatever checkpoint load_saved_model picks, so the offline steps address the
# right directories without hard-coding an epoch.
EPOCH=$(python - "$MODEL_DIR" <<'PY'
import glob, os, re, sys
d = sys.argv[1]
best = glob.glob(os.path.join(d, 'net_epoch_bestval_at*.pth'))
if best:
    print(re.findall(r'at(\d+)', os.path.basename(best[0]))[0])
else:
    eps = [int(re.findall(r'epoch(\d+)', os.path.basename(p))[0])
           for p in glob.glob(os.path.join(d, 'net_epoch*.pth'))]
    print(max(eps) if eps else 0)
PY
)
SUF="${RANGE%,*}_${RANGE#*,}_epoch${EPOCH}"
echo "== model_dir=$MODEL_DIR  epoch=$EPOCH  artefact suffix=$SUF"
mkdir -p "$OUT"

# --------------------------------------------------------------------------- #
# 1. GPU passes. One per condition; --levy_diag writes the per-box dump that
#    every offline step below reads. Pose noise is zero-mean Gaussian on each
#    agent's x/y/yaw, seeded per dataloader worker so a level is reproducible.
# --------------------------------------------------------------------------- #
echo "== [1/3] clean reference (no diagnostic, the headline AP/FP numbers)"
$INFER

echo "== [2/3] trust-map statistics + per-box dump, clean"
$INFER --levy_diag --save_levy --note _diag

echo "== [3/3] pose-noise sweep, with the trust-map variants"
#            pos_std  rot_std  tag
for lvl in  "0.0 0.0 var" "0.2 0.2 varn02" "0.5 0.5 varn05" \
            "0.8 1.0 varn08" "1.0 1.0 varn10"; do
  set -- $lvl
  if [ "$1" = "0.0" ]; then
    $INFER --levy_diag --note "_$3"                       # clean, with variants
  else
    $INFER --levy_diag --pos_std "$1" --rot_std "$2" --note "_$3"
  fi
done

# --------------------------------------------------------------------------- #
# 2. Offline analysis. No GPU. Protocols 1 and 3, the variant channels, the
#    evidence-gate grid and the per-channel beta sweep all come from here.
# --------------------------------------------------------------------------- #
echo "== offline: benchmark harness per condition"
for tag in diag var varn02 varn05 varn08 varn10; do
  npz="$MODEL_DIR/levy_intermediate_${tag}_${SUF}/levy_diag_boxes.npz"
  [ -f "$npz" ] || { echo "   skip $tag (no dump)"; continue; }
  python opencood/tools/levy_fp_benchmark.py --npz "$npz" --out_dir "$OUT" --tag "$tag"
done

echo "== offline: drop-filter sweep, raw and corrected channels"
NPZ_CLEAN="$MODEL_DIR/levy_intermediate_var_${SUF}/levy_diag_boxes.npz"
for ch in sigma cos_shift.sigma cos.disagreement; do
  echo "-- channel $ch"
  python opencood/tools/levy_drop_sweep.py --npz "$NPZ_CLEAN" \
      --channel "$ch" --pct 1,5,10,25
done

# --------------------------------------------------------------------------- #
# 3. The shipped rule, applied end to end so the gain is measured by the real
#    evaluator rather than replayed. Compare its eval yaml against the clean
#    reference from step 1.
# --------------------------------------------------------------------------- #
echo "== shipped rule: evidence gate + cosine disagreement penalty"
$INFER --levy_evidence_gate 0.5 \
       --levy_disagreement_channel cos \
       --levy_disagreement_beta 0.75 \
       --levy_disagreement_norm 0.2633 \
       --note _shipped

echo
echo "== done. eval_*.yaml in $MODEL_DIR, JSON + plots in $OUT"
echo "   baseline vs shipped:"
grep -H -E '^(ap_70|fp_rate_50)' \
    "$MODEL_DIR/eval_intermediate_${SUF}.yaml" \
    "$MODEL_DIR/eval_intermediate_shipped_${SUF}.yaml" 2>/dev/null || true
