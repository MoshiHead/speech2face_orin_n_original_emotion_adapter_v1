"""Extract per-80ms Helium hidden states for the emotion dataset by
teacher-forcing the PersonaPlex trunk's AGENT stream with the emotional
reference audio (moshi_tokens in LMGen.prepare_step_input).

This is the exact reply path the live server feeds the motion student from:
same engine class (MoshiOnlyEngineWithHidden), same Robert_5 voice + persona
text prompt, same int4 trunk conversion, same graphed _step with the layer
hidden capture. The USER stream is held at silence, matching the live
listening state while the agent speaks.

Resumable: clips whose output .pt already exists are skipped.
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

AUDIO_EXTS = ("wav", "mp3", "flac", "ogg", "m4a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--motion_dir", required=True)
    ap.add_argument("--raw_audio_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--voice", default="Robert_5.pt")
    ap.add_argument("--text_prompt", required=True)
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--reset_mode", default="emotion",
                    choices=["emotion", "clip", "never"])
    ap.add_argument("--int4_g", type=int, default=32, help="0 = bf16 trunk")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    src = repo / "src"
    os.environ.setdefault("IMTALKER_DIR", str(src / "imtalker"))
    os.environ.setdefault("DESIGN_A_DIR", str(src / "niloys"))
    for p in (str(src / "imtalker"), str(src / "moshi_pkg"), str(src / "niloys"),
              str(src / "niloys" / "live"), str(src / "niloys" / "emotion")):
        if p not in sys.path:
            sys.path.insert(0, p)

    weights = repo / "weights"
    snap = Path((repo / ".snap_path").read_text().strip())
    moshi_weight = snap / "model.safetensors"
    mimi_weight = weights / "mimi.safetensors"
    tokenizer = weights / "tokenizer_spm_32k_3.model"
    for f in (moshi_weight, mimi_weight, tokenizer):
        assert f.exists(), "missing %s - run the Section 0 setup cells (or prepare_x86.sh) first" % f

    import torch

    if args.int4_g > 0:
        # identical conversion to the live server (the try19 int4 loader hook)
        from moshi.models import loaders as _loaders
        from int4_linear import convert_bf16 as _conv_int4
        _orig_get_lm = _loaders.get_moshi_lm

        def _get_moshi_lm_int4(*a, **k):
            lm = _orig_get_lm(*a, **k)
            n = _conv_int4(lm.transformer, args.int4_g)
            torch.cuda.empty_cache()
            print("[extract] int4 G=%d: %d Linears converted" % (args.int4_g, n),
                  flush=True)
            return lm
        _loaders.get_moshi_lm = _get_moshi_lm_int4
        print("[extract] int4 loader hook installed (matches the live server)",
              flush=True)
    else:
        print("[extract] bf16 trunk", flush=True)

    from imtalker_native_passthrough import (
        MoshiOnlyEngineWithHidden, load_audio_24k)
    from emotion_student import EMOTION_NAMES, parse_emotion_from_stem

    engine = MoshiOnlyEngineWithHidden(
        moshi_root=str(weights),
        mimi_hf_repo="nvidia/personaplex-7b-v1",
        device="cuda",
        cfg_coef=1.0,
        placeholder_jpeg_b64="",
        moshi_weight=str(moshi_weight),
        mimi_weight=str(mimi_weight),
        tokenizer=str(tokenizer),
        quantize_4bit=False,
        num_codebooks=8,
        context=None,
        voice_prompt=args.voice,
        voice_prompt_dir=str(weights / "voices"),
        text_prompt=args.text_prompt,
    )
    mimi = engine.mimi
    FRAME = int(mimi.sample_rate / mimi.frame_rate)     # 1920 samples = 80 ms
    device = engine.device

    # Full-clip bulk encodes hit the encoder transformer with lengths its CUDA
    # graph was never captured for (CUDAGraphed is shape-static) - bypass it,
    # exactly like the live server's raw voice prompt path. Re-applied after
    # every streaming reset in case the state (and its graph) is re-created.
    def _disable_mimi_enc_graph():
        st = getattr(mimi, "_streaming_state", None)
        g = getattr(st, "graphed_tr_enc", None)
        if g is not None:
            g.disable = True

    # one fixed silence code column for the USER stream (encoded once, reused)
    mimi.reset_streaming()
    _disable_mimi_enc_graph()
    with torch.no_grad():
        silence = mimi.encode(torch.zeros(1, 1, FRAME, device=device))[:, :8].clone()

    # ---- clip list: motion stems that also have raw audio + an emotion label
    motion_files = sorted(glob.glob(os.path.join(args.motion_dir, "*.pt")))
    audio = {}
    for ext in AUDIO_EXTS:
        for f in glob.glob(os.path.join(args.raw_audio_dir, "**", "*." + ext),
                           recursive=True):
            audio.setdefault(Path(f).stem, f)
    clips, n_no_audio, n_no_emo = [], 0, 0
    for mf in motion_files:
        stem = Path(mf).stem
        eid = parse_emotion_from_stem(stem)
        if stem not in audio:
            n_no_audio += 1
            continue
        if eid is None:
            n_no_emo += 1
            continue
        clips.append((stem, eid, mf, audio[stem]))
    clips.sort(key=lambda c: (c[1], c[0]))   # group by emotion: fewer voice-prompt resets
    if args.max_clips:
        clips = clips[: args.max_clips]
    print("[extract] %d clips (%d skipped: no audio, %d skipped: no emotion label)"
          % (len(clips), n_no_audio, n_no_emo), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"clips": [], "frame_samples": FRAME, "sample_rate": 24000,
                "voice": args.voice, "text_prompt": args.text_prompt,
                "int4_g": args.int4_g, "reset_mode": args.reset_mode,
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    last_emotion, t_start = None, time.time()
    for i, (stem, eid, mf, af) in enumerate(clips):
        out_path = os.path.join(args.out_dir, stem + ".pt")
        if os.path.exists(out_path):
            continue                                    # resumable
        wav_np = load_audio_24k(af)                     # float32 mono @ 24 kHz
        motion = torch.load(mf, map_location="cpu", weights_only=False)
        if isinstance(motion, dict):
            motion = motion.get("motion", motion.get("motion_latents"))
        n_steps = min(int(len(wav_np)) // FRAME, int(motion.shape[0]) // 2)
        if n_steps < 4:
            print("[extract] skip %s: too short" % stem, flush=True)
            continue
        wav = torch.from_numpy(wav_np[: n_steps * FRAME].copy()).view(1, 1, -1).to(device)
        mimi.reset_streaming()
        _disable_mimi_enc_graph()
        with torch.no_grad():
            codes = mimi.encode(wav)[:, :8]             # [1,8,T] agent-stream codes
        if (args.reset_mode == "clip"
                or (args.reset_mode == "emotion" and eid != last_emotion)):
            engine.reset_session()                      # re-apply Robert_5 + persona
        last_emotion = eid
        hidden = []
        with torch.no_grad():
            for t in range(n_steps):
                out = engine.lm_gen._step(input_tokens=silence,
                                          moshi_tokens=codes[:, :, t:t + 1])
                if out is None:                         # first call after a reset
                    continue
                hidden.append(out[2][:1, -1:].detach().float().cpu())
        h = torch.cat(hidden, 1)[0].contiguous()        # [T,4096]
        torch.save({"helium": h, "emotion_id": eid, "stem": stem}, out_path)
        manifest["clips"].append({"stem": stem, "emotion_id": eid,
                                  "emotion": EMOTION_NAMES[eid],
                                  "steps": int(h.shape[0])})
        el = time.time() - t_start
        print("[extract] %4d/%4d %-42s %-9s T=%4d  %5.1fs/clip  elapsed %5.1f min"
              % (i + 1, len(clips), stem, EMOTION_NAMES[eid], h.shape[0],
                 el / (i + 1), el / 60), flush=True)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("EXTRACTION DONE: %d clips -> %s" % (len(manifest["clips"]), args.out_dir),
          flush=True)


if __name__ == "__main__":
    main()
