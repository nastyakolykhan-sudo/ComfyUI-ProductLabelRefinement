#!/bin/zsh
# check_all.sh: every regression of this pack in one run, against a private copy of the skill carrying this checkout's
# engine (the shared skill is never touched). Run it before every merge and after tools/sync_engine.py --update.
#
#   tools/check_all.sh [--skip-all]        # STILLS, VIDEO, ALL (one workflow on a CPU ComfyUI, ~1-3 h), FIXTURES
#
#   BRAND     tools/brand_check.py --all     no client or brand names in this public pack (terms: PLR_BRAND_TERMS)
#   STILLS    tools/regress.py --psd          the stills nodes against the skill's run.py (both call imgpost.stages)
#   VIDEO     tools/regress_video.py          Refine (Video) against the skill's video.py --refine
#   ALL       tools/regress_all.py --video    the one workflow against the separate templates, on a live server
#   FIXTURES  scripts/video_regress.py        the video engine's 8 fixtures, 0 px
#
# Environment (defaults for this Mac): PLR_COMFY (ComfyUI test install with venv/ and start-plr.sh), PLR_ALLTRACKER
# (weights), PLR_PORT (8211), PLR_SKILL (default: imgpost.lock.json's "skill"), PLR_CHECKS_OUT (logs; default
# ~/Documents/ComfyUI/plr-checks/<branch>), PLR_HEAVY_LOCK (one heavy local job at a time across sessions), PLR_BRAND_TERMS (the
# private terms file, default ~/.config/silverside/brand-terms.txt; never in a repo).
# Exit 0 only when every part passes; the summary is in $OUT/summary.txt.
set -u
P=${0:A:h:h}
T=${PLR_COMFY:-~/Documents/ComfyUI/comfyui-test}; PY=$T/venv/bin/python
W=${PLR_ALLTRACKER:-~/Documents/ComfyUI/models/trackers/alltracker/alltracker.pth}
PORT=${PLR_PORT:-8211}
SKILL=${PLR_SKILL:-$(python3 -c "import json,os;print(os.path.expanduser(json.load(open('$P/imgpost.lock.json'))['skill']))")}
BR=$(git -C $P rev-parse --abbrev-ref HEAD | tr '/' '-')
OUT=${PLR_CHECKS_OUT:-~/Documents/ComfyUI/plr-checks/$BR}; K=$OUT/skill-copy
LOCK=${PLR_HEAVY_LOCK:-"$HOME/Library/Application Support/Claude/side-session-notes/local_196b10b9-031b-4ca3-952d-7fb63c8e38cc/local-heavy-job.lock"}
TAG="check_all $BR ($P)"
if [[ -f "$LOCK" ]] && ! grep -qF "$TAG" "$LOCK"; then echo "another local heavy job runs: $(cat "$LOCK")"; exit 3; fi
mkdir -p "${LOCK:h}" $OUT; echo "$TAG, port $PORT, started $(date +%T)" > "$LOCK"
trap 'rm -f "$LOCK"; pkill -f "main.py.*--port $PORT" 2>/dev/null' EXIT

# the skill copy: the shared skill with this checkout's engine
rm -rf $K; mkdir -p $K; rsync -a --exclude __pycache__ --exclude .DS_Store $SKILL/ $K/
for pkg in imgpost imgvideo imgauto; do rsync -a --delete --exclude __pycache__ --exclude .DS_Store $P/$pkg/ $K/scripts/$pkg/; done
S=$OUT/summary.txt; echo "pack $(git -C $P rev-parse --short HEAD)$(git -C $P diff --quiet || echo ' + uncommitted changes') ($P), skill copy $K" > $S
rc=0; step() { echo "$1 $2" >> $S; [[ $2 == 0 ]] || rc=1; }

cd $P
python3 -P tools/brand_check.py --all --terms ${PLR_BRAND_TERMS:-~/.config/silverside/brand-terms.txt} > $OUT/brand.log 2>&1; step BRAND $?
$PY -P tools/regress.py --psd --skill-dir $K > $OUT/stills.log 2>&1; step STILLS $?
$PY -P tools/regress_video.py --comfy $T --weights $W --skill-dir $K > $OUT/video.log 2>&1; step VIDEO $?
if [[ ${1:-} != --skip-all ]]; then
  pkill -f "main.py.*--port $PORT" 2>/dev/null
  mkdir -p $T/instances/$PORT/input; cp -n $T/input/example.png $T/input/inkdemo_packshot.png $T/instances/$PORT/input/ 2>/dev/null
  $T/start-plr.sh $PORT $P > $OUT/server.log 2>&1 &
  for i in {1..60}; do curl -s -o /dev/null http://127.0.0.1:$PORT/system_stats && break; sleep 5; done
  $PY -P tools/regress_all.py --server http://127.0.0.1:$PORT --video --timeout 10800 > $OUT/all.log 2>&1; step ALL $?
  pkill -f "main.py.*--port $PORT"
else
  echo "ALL skipped" >> $S
fi
(cd $K/scripts && PLR_ALLTRACKER=$W $PY -P video_regress.py > $OUT/fixtures.log 2>&1); step FIXTURES $?
cat $S
exit $rc
