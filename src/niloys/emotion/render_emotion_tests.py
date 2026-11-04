"""Render one emotional clip through the emotion student for every emotion id,
with the distilled single-identity renderer - the offline equivalent of the old
pipeline's test_emotions.py. Free-running rollout (no teacher forcing), the
same code path the live server takes, so these videos predict live behaviour.
"""
import argparse, os, subprocess, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--base_student", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--helium_dir", required=True)
    ap.add_argument("--raw_audio_dir", required=True)
    ap.add_argument("--clip_stem", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gain", type=float, default=1.2,
                    help="same motion gain the live server applies")
    ap.add_argument("--max_seconds", type=float, default=12.0)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    src = repo / "src"
    os.environ.setdefault("DESIGN_A_DIR", str(src / "niloys"))
    for p in (str(src / "niloys"), str(src / "niloys" / "emotion")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from student.renderer import build as build_renderer
    from emotion_student import EMOTION_NAMES, NUM_EMOTIONS, build_emotion_student

    device = "cuda"
    model, cfg, meta = build_emotion_student(args.base_student, args.adapter, device)
    print("[render] student + adapter loaded (trained step %s)"
          % meta.get("step"), flush=True)

    rpack = torch.load(repo / "weights" / "renderer_narrow.pt",
                       map_location="cpu", weights_only=False)
    R = build_renderer(**rpack["cfg"]).to(device).float().eval()
    R.load_state_dict(rpack["model"])
    for p in R.parameters():
        p.requires_grad_(False)
    print("[render] distilled renderer loaded (val_PSNR=%.2f dB)"
          % rpack.get("val_psnr", 0.0), flush=True)

    hpack = torch.load(os.path.join(args.helium_dir, args.clip_stem + ".pt"),
                       map_location="cpu", weights_only=False)
    h = hpack["helium"] if isinstance(hpack, dict) else hpack
    T = min(h.shape[0] - cfg.lookahead, int(args.max_seconds * 12.5))
    print("[render] clip %s: %d steps (%.1f s)" % (args.clip_stem, T, T / 12.5),
          flush=True)

    wav_path = None
    for ext in ("wav", "mp3", "flac", "ogg", "m4a"):
        c = os.path.join(args.raw_audio_dir, args.clip_stem + "." + ext)
        if os.path.exists(c):
            wav_path = c
            break
    import cv2
    import numpy as np

    os.makedirs(args.out_dir, exist_ok=True)
    motions = {}
    for eid in range(NUM_EMOTIONS):
        model.set_emotion(eid)
        st = model.init_state(1, device, torch.float32)
        outs = []
        with torch.no_grad():
            for t in range(T):
                o = model.step(h[t + cfg.lookahead: t + cfg.lookahead + 1].to(device), st)
                outs.append(o[0])
        pred_n = torch.stack(outs, 0)                       # [T,2,32] normalized
        if args.gain != 1.0:
            mu = pred_n.mean(dim=0, keepdim=True)
            pred_n = mu + (pred_n - mu) * args.gain
        motion = model.denorm_motion(pred_n).reshape(-1, cfg.motion_dim).float().cpu()
        motions[eid] = motion

        frames = []
        with torch.no_grad():
            for i in range(0, motion.shape[0], 16):
                mb = motion[i: i + 16].to(device)
                img = R(mb)                                 # [B,3,H,W] in [0,1]
                frames.append((img.permute(0, 2, 3, 1) * 255).byte().cpu().numpy())
        frames = np.concatenate(frames, 0)[:, :, :, ::-1]   # RGB -> BGR
        H, W = frames.shape[1:3]
        tmp = os.path.join(args.out_dir, "_tmp_%d.mp4" % eid)
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
        for fr in frames:
            vw.write(fr)
        vw.release()
        out_path = os.path.join(args.out_dir, "%d_%s.mp4" % (eid, EMOTION_NAMES[eid]))
        if wav_path:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                            "-i", wav_path, "-c:v", "copy", "-c:a", "aac",
                            "-shortest", out_path], check=True)
            os.remove(tmp)
        else:
            os.replace(tmp, out_path)
        print("[render] %s -> %s (%d frames)"
              % (EMOTION_NAMES[eid], out_path, len(frames)), flush=True)

    ref = motions[0]
    print("[render] mean |motion - neutral| per emotion (raw units):", flush=True)
    for eid in range(1, NUM_EMOTIONS):
        d = (motions[eid] - ref).abs().mean().item()
        print("[render]   %-9s %.4f" % (EMOTION_NAMES[eid], d), flush=True)
    print("RENDER DONE -> %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
