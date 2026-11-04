"""Emotion-aware variant of imtalker_try19_student_live.py.

Every piece of the verified try19 server (int4 trunk, one-pass depformer,
distilled renderer, TensorRT / GAN Mimi decoder, CUDA-graphed student stepping)
is reused untouched. The only changes, applied by monkey-patching try19's
module globals before try19.main() runs:

  * load_student        -> loads the base student, attaches the emotion
                           conditioner, merges the emotion LoRA adapter into
                           the base weights, and sets the startup emotion
                           (EMOTION_ID).
  * make_student_sampler -> wraps the returned sampler so each 80 ms chunk
                           first polls EMOTION_CONTROL_FILE: writing an emotion
                           id (0-5) or name to that file switches the avatar's
                           emotion mid-conversation. The switch is an in-place
                           update of the emotion_bias buffer, which the captured
                           student CUDA graph keeps reading - no re-capture.

Env vars:
  EMOTION_LORA_CKPT     adapter checkpoint written by train_emotion_lora.ipynb
  EMOTION_ID            startup emotion: id 0-5 or name (default 1 = happy)
  EMOTION_CONTROL_FILE  runtime control file ("" disables runtime switching)
"""
import os
import sys
from pathlib import Path

LIVE_DIR = Path(__file__).resolve().parent
EMOTION_DIR = LIVE_DIR.parents[0] / "emotion"
for _p in (str(LIVE_DIR), str(EMOTION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import imtalker_try19_student_live as try19                            # noqa: E402
from emotion_student import (                                          # noqa: E402
    EMOTION_MAP, EMOTION_NAMES, build_emotion_student)

ADAPTER = os.environ.get("EMOTION_LORA_CKPT", "").strip()
CONTROL = os.environ.get(
    "EMOTION_CONTROL_FILE",
    str(LIVE_DIR.parents[2] / "emotion_control.txt")).strip()


def _emotion_index(v):
    v = str(v).strip().lower()
    if v.lstrip("-").isdigit():
        i = int(v)
        return i if 0 <= i < len(EMOTION_NAMES) else None
    return EMOTION_MAP.get(v)


STARTUP = _emotion_index(os.environ.get("EMOTION_ID", "1"))
if STARTUP is None:
    STARTUP = 1

def _load_emotion_student(ckpt, device):
    """try19.load_student replacement: base student + emotion adapter."""
    if not ADAPTER or not os.path.exists(ADAPTER):
        raise SystemExit(
            "[emotion] EMOTION_LORA_CKPT is not set or the file is missing: %r\n"
            "[emotion] train an adapter with train_emotion_lora.ipynb first "
            "(or set EMOTION_HF_REPO in emotion_live_streaming.ipynb)" % ADAPTER)
    model, cfg, meta = build_emotion_student(ckpt, ADAPTER, device)
    model.set_emotion(STARTUP)
    print("[emotion] adapter: %s (trained step %s, LoRA rank %s)"
          % (ADAPTER, meta.get("step"), meta.get("lora_rank")), flush=True)
    print("[emotion] startup emotion: %d (%s); runtime control file: %s"
          % (STARTUP, EMOTION_NAMES[STARTUP], CONTROL or "(disabled)"), flush=True)
    return model, cfg


try19.load_student = _load_emotion_student

_orig_make_student_sampler = try19.make_student_sampler


def _make_emotion_sampler(model, cfg, **kw):
    """Wrap try19's sampler: poll the control file once per chunk, then run the
    unmodified student sampler (KV state, blending, gain, CUDA graph)."""
    inner = _orig_make_student_sampler(model, cfg, **kw)
    state = {"mtime": None, "idx": None}

    def _poll():
        if not CONTROL:
            return
        try:
            mtime = os.path.getmtime(CONTROL)
        except OSError:
            return
        if mtime == state["mtime"]:
            return
        state["mtime"] = mtime
        try:
            idx = _emotion_index(open(CONTROL).read())
        except Exception:
            return
        if idx is None or idx == state["idx"]:
            return
        state["idx"] = idx
        model.set_emotion(idx)          # in-place buffer update: graph-safe
        print("[emotion] switched to %d (%s)" % (idx, EMOTION_NAMES[idx]),
              flush=True)

    def sampler(self, helium, target_frames):
        _poll()
        return inner(self, helium, target_frames)

    return sampler


try19.make_student_sampler = _make_emotion_sampler

if __name__ == "__main__":
    try19.main()
