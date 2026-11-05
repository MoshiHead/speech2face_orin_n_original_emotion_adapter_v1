"""Render one clip through the emotion student for every emotion id, with the
distilled single-identity renderer - the offline equivalent of the old
pipeline's test_emotions.py. Free-running rollout (no teacher forcing), the
same code path the live server takes, so these videos predict live behaviour.

Motion-range handling (the white-video fix): the distilled renderer predicts a
residual on a baked-in base frame, conditioned via FiLM on
(motion - R.motion_mean) / R.motion_std. It was distilled on ONE identity's
motion distribution (Robert); emotional datasets sit far outside it, which
saturates the FiLM conditioning and clamps every frame to white. The default
--motion_map affine standardizes the clip's motion per dimension onto the
renderer stats (within-clip dynamics - the emotion signal - are preserved;
only the identity-level pose/scale is remapped onto Robert), followed by a
safety clamp at renderer mean +/- 4 sigma.
"""
import argparse, os, subprocess, sys
from pathlib import Path

FPS = 25                      # 2 motion frames per 80 ms step


def _mux_audio(tmp, wav_path, out_path):
    if wav_path:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                        "-i", wav_path, "-c:v", "copy", "-c:a", "aac",
                        "-shortest", out_path], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, out_path)


def find_audio(raw_audio_dir, stem):
    for ext in ("wav", "mp3", "flac", "ogg", "m4a"):
        c = os.path.join(raw_audio_dir, stem + "." + ext)
        if os.path.exists(c):
            return c
    return None


def load_renderer(repo_root, device):
    import torch
    from student.renderer import build as build_renderer
    rpack = torch.load(Path(repo_root) / "weights" / "renderer_narrow.pt",
                       map_location="cpu", weights_only=False)
    R = build_renderer(**rpack["cfg"]).to(device).float().eval()
    R.load_state_dict(rpack["model"])
    for p in R.parameters():
        p.requires_grad_(False)
    return R, rpack


def rollout_motion(model, helium, cfg, emotion_id, device, gain=1.2,
                   max_seconds=12.0):
    """Free-running rollout -> [2T, 32] RAW (denormalized) motion on CPU."""
    import torch
    T = min(helium.shape[0] - cfg.lookahead, int(max_seconds * 12.5))
    model.set_emotion(emotion_id)
    st = model.init_state(1, device, torch.float32)
    outs = []
    with torch.no_grad():
        for t in range(T):
            o = model.step(helium[t + cfg.lookahead: t + cfg.lookahead + 1]
                           .to(device), st)
            outs.append(o[0])
    pred_n = torch.stack(outs, 0)                          # [T,2,32] normalized
    if gain != 1.0:
        mu = pred_n.mean(dim=0, keepdim=True)
        pred_n = mu + (pred_n - mu) * gain
    return model.denorm_motion(pred_n).reshape(-1, cfg.motion_dim).float().cpu()


def map_motion_for_renderer(motion, R, mode="affine", clamp_sigmas=4.0):
    """Bring raw [N,32] motion into the distilled renderer's training range.

    'affine': standardize the clip per dimension onto the renderer stats.
    'clamp' : only clip to renderer mean +/- clamp_sigmas * std.
    'none'  : leave the motion untouched (diagnostic).
    Returns (mapped_motion, stats_string).
    """
    import torch
    mean, std = R.motion_mean.cpu(), R.motion_std.cpu().clamp_min(1e-6)
    lo, hi = mean - clamp_sigmas * std, mean + clamp_sigmas * std
    raw = motion.clone()
    if mode == "affine":
        cm, cs = motion.mean(0), motion.std(0).clamp_min(1e-6)
        motion = (motion - cm) / cs * std + mean
    clipped = ((motion < lo) | (motion > hi)).float().mean().item()
    motion = torch.maximum(torch.minimum(motion, hi), lo)
    stats = ("raw range [%+.3f, %+.3f]  renderer sigma range [%+.1f, %+.1f]  "
             "clamped %.1f%% of values"
             % (raw.min().item(), raw.max().item(),
                ((raw - mean) / std).min().item(),
                ((raw - mean) / std).max().item(), 100 * clipped))
    return motion, stats


def render_video_frames(R, motion, device, out_prefix=None):
    """raw [N,32] motion -> uint8 BGR frames [N,H,W,3]; dumps the first frame."""
    import numpy as np
    import torch
    frames = []
    with torch.no_grad():
        for i in range(0, motion.shape[0], 16):
            img = R(motion[i: i + 16].to(device))          # [B,3,H,W] in [0,1]
            frames.append((img.permute(0, 2, 3, 1) * 255).byte().cpu().numpy())
    frames = np.concatenate(frames, 0)[:, :, :, ::-1]      # RGB -> BGR
    if out_prefix:
        import cv2
        cv2.imwrite(out_prefix + "_first_frame.png", frames[0])
    return frames


def write_video(frames, wav_path, out_path, fps=FPS):
    import cv2
    H, W = frames.shape[1:3]
    tmp = out_path + ".silent.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fr in frames:
        vw.write(fr)
    vw.release()
    _mux_audio(tmp, wav_path, out_path)
    return out_path


def render_motion_video(model, R, helium, cfg, emotion_id, wav_path, out_path,
                        device, gain=1.2, max_seconds=12.0, motion_map="affine"):
    """rollout -> range map -> frames -> mp4 (audio muxed).

    Shared by this CLI and by the periodic test inside train_emotion.py.
    Returns (raw_motion, stats_string, n_frames).
    """
    motion = rollout_motion(model, helium, cfg, emotion_id, device,
                            gain=gain, max_seconds=max_seconds)
    mapped, stats = map_motion_for_renderer(motion, R, motion_map)
    frames = render_video_frames(R, mapped, device,
                                 out_prefix=os.path.splitext(out_path)[0])
    write_video(frames, wav_path, out_path)
    return motion, stats, frames.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--base_student", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--helium_dir", required=True)
    ap.add_argument("--raw_audio_dir", required=True)
    ap.add_argument("--clip_stem", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--motion_dir", default="",
                    help="dataset motion dir -> also renders the ground-truth "
                         "motion diagnostic videos (gt_none/gt_affine)")
    ap.add_argument("--gain", type=float, default=1.2,
                    help="same motion gain the live server applies")
    ap.add_argument("--max_seconds", type=float, default=12.0)
    ap.add_argument("--motion_map", default="affine",
                    choices=["affine", "clamp", "none"])
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    src = repo / "src"
    os.environ.setdefault("DESIGN_A_DIR", str(src / "niloys"))
    for p in (str(src / "niloys"), str(src / "niloys" / "emotion")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from emotion_student import EMOTION_NAMES, NUM_EMOTIONS, build_emotion_student

    device = "cuda"
    model, cfg, meta = build_emotion_student(args.base_student, args.adapter, device)
    print("[render] student + adapter loaded (trained step %s)"
          % meta.get("step"), flush=True)

    R, rpack = load_renderer(repo, device)
    print("[render] distilled renderer loaded (val_PSNR=%.2f dB)"
          % rpack.get("val_psnr", 0.0), flush=True)

    # ---- range diagnostics: student stats vs the renderer's expectations ----
    rmean, rstd = R.motion_mean.cpu(), R.motion_std.cpu()
    smean, sstd = model.motion_mean.cpu(), model.motion_std.cpu()
    smean, sstd = model.motion_mean.cpu(), model.motion_std.cpu()
    print("[render] student  motion stats: mean |%.3f|  std %.3f"
          % (float(smean.abs().mean()), float(sstd.mean())), flush=True)
    print("[render] renderer motion stats: mean |%.3f|  std %.3f"
          % (float(rmean.abs().mean()), float(rstd.mean())), flush=True)

    hpack = torch.load(os.path.join(args.helium_dir, args.clip_stem + ".pt"),
                       map_location="cpu", weights_only=False)
    h = (hpack["helium"] if isinstance(hpack, dict) else hpack).float()
    T = min(h.shape[0] - cfg.lookahead, int(args.max_seconds * 12.5))
    print("[render] clip %s: %d steps (%.1f s)"
          % (args.clip_stem, T, T / 12.5), flush=True)

    wav_path = find_audio(args.raw_audio_dir, args.clip_stem)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- ground-truth diagnostic: the DATASET's motion through the renderer.
    # If gt_none is white but gt_affine shows a talking face, the white videos
    # are a motion-range problem (OOD w.r.t. Robert), independent of the
    # student, the extraction and the training.
    if args.motion_dir:
        mp = os.path.join(args.motion_dir, args.clip_stem + ".pt")
        if os.path.exists(mp):
            m = torch.load(mp, map_location="cpu", weights_only=False)
            if isinstance(m, dict):
                m = m.get("motion", m.get("motion_latents"))
            gt = m.float()[: 2 * T]
            sig = (gt - rmean) / rstd.clamp_min(1e-6)
            print("[render] GT motion range [%+.3f, %+.3f]  sigma range [%+.1f, %+.1f]"
                  % (float(gt.min()), float(gt.max()),
                     float(sig.min()), float(sig.max())), flush=True)
            for mode in ("none", "affine"):
                m2, _st = map_motion_for_renderer(gt.clone(), R, mode)
                frames = render_video_frames(R, m2, device,
                                             out_prefix=os.path.join(
                                                 args.out_dir, "gt_%s" % mode))
                out = os.path.join(args.out_dir, "gt_%s_%s.mp4"
                                   % (mode, args.clip_stem))
                write_video(frames, wav_path, out)
                print("[render] GT %s -> %s" % (mode, out), flush=True)

    motions = {}
    for eid in range(NUM_EMOTIONS):
        out_path = os.path.join(args.out_dir, "%d_%s.mp4" % (eid, EMOTION_NAMES[eid]))
        motion, stats, nfr = render_motion_video(
            model, R, h, cfg, eid, wav_path, out_path, device,
            gain=args.gain, max_seconds=args.max_seconds,
            motion_map=args.motion_map)
        motions[eid] = motion
        print("[render] %-8s -> %s (%d frames)"
              % (EMOTION_NAMES[eid], out_path, nfr), flush=True)
        print("[render]   motion[%s]  %s" % (EMOTION_NAMES[eid], stats), flush=True)

    ref = motions[0]
    print("[render] mean |motion - neutral| per emotion (raw units):", flush=True)
    for eid in range(1, NUM_EMOTIONS):
        d = (motions[eid] - ref).abs().mean().item()
        print("[render]   %-9s %.4f" % (EMOTION_NAMES[eid], d), flush=True)
    print("RENDER DONE -> %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()

