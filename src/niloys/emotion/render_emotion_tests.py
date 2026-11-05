"""Render one clip through the emotion student for every emotion id.

Two renderers, selected with --renderer:

  * imt (default): the vendored reference-image-conditioned IMTalker renderer
    from the old pipeline (imt_renderer/). It consumes the SAME 32-D motion
    tokens the student predicts - the student was distilled from the old FM
    generator whose sample() output fed exactly this renderer - and takes an
    arbitrary reference image. This is the old pipeline's test_emotions.py
    path (ref image + audio + emotion -> video), restored.

  * distilled: the single-identity 512x512 residual renderer the live server
    uses. Kept for live-parity checks and for --diag.

--diag runs the staged white-screen investigation of the DISTILLED renderer:
  1. the renderer's own mean motion (must reproduce the base face if the
     checkpoint/buffers are sane)
  2. mean +/- k*std sweeps (maps the FiLM saturation boundary)
  3. dataset ground-truth motion, raw and affine-mapped
  4. student rollout motion, raw and affine-mapped
printing base-frame / motion / brightness stats at every stage and saving one
PNG per stage, so the exact stage where frames turn white is visible.
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


# ---------------------------------------------------------------------------
# IMT renderer (reference-image conditioned; the old pipeline's renderer)
# ---------------------------------------------------------------------------
def load_imt_renderer(imt_ckpt, ref_image, device):
    import argparse, cv2, torch
    from imt_renderer.models import IMTRenderer
    opt = argparse.Namespace(num_heads=8, swin_res_threshold=128, window_size=8)
    R = IMTRenderer(opt).to(device).float().eval()
    ck = torch.load(imt_ckpt, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck)
    state = {k[4:]: v for k, v in state.items() if k.startswith("gen.")}
    missing, unexpected = R.load_state_dict(state, strict=False)
    for p in R.parameters():
        p.requires_grad_(False)
    print("[imt] renderer: %s (missing=%d unexpected=%d)"
          % (imt_ckpt, len(missing), len(unexpected)), flush=True)
    # ref image -> identity + motion-ref features (the old pipeline's
    # DataProcessor: RGB -> resize 512 -> tensor in [0,1], crop=False)
    img = cv2.imread(ref_image, cv2.IMREAD_COLOR)
    assert img is not None, "reference image not found: %s" % ref_image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    s = torch.from_numpy(img.copy()).float().permute(2, 0, 1) \
        .unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        f_r, g_r = R.dense_feature_encoder(s)
        t_r = R.latent_token_encoder(s)
        ta_r = R.adapt(t_r, g_r)
        m_r = R.latent_token_decoder(ta_r)
    print("[imt] reference: %s -> f_r x%d, g_r %s, m_r x%d"
          % (ref_image, len(f_r), tuple(g_r.shape), len(m_r)), flush=True)
    return {"R": R, "f_r": f_r, "g_r": g_r, "m_r": m_r}


def _exp(seq, n):
    return [t.expand(n, *([-1] * (t.ndim - 1))) for t in seq]


def render_frames_imt(ctx, motion, device, out_prefix=None, batch=8):
    """raw [N,32] motion tokens -> uint8 BGR frames [N,512,512,3]."""
    import cv2, numpy as np, torch
    R, f_r, g_r, m_r = ctx["R"], ctx["f_r"], ctx["g_r"], ctx["m_r"]
    frames = []
    with torch.no_grad():
        for i in range(0, motion.shape[0], batch):
            mo = motion[i: i + batch].to(device)
            B = mo.shape[0]
            ta_c = R.adapt(mo, g_r.expand(B, -1))
            m_c = R.latent_token_decoder(ta_c)
            img = R.decode(m_c, _exp(m_r, B), _exp(f_r, B))      # [B,3,512,512]
            frames.append((img.clamp(0, 1).permute(0, 2, 3, 1)
                           * 255).byte().cpu().numpy())
    frames = np.concatenate(frames, 0)[:, :, :, ::-1]            # RGB -> BGR
    if out_prefix:
        cv2.imwrite(out_prefix + "_first_frame.png", frames[0])
    return frames


# ---------------------------------------------------------------------------
# Distilled single-identity renderer (the live server's renderer)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# shared: rollout + dispatching render
# ---------------------------------------------------------------------------
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


def render_motion_video(model, renderer, helium, cfg, emotion_id, wav_path,
                        out_path, device, gain=1.2, max_seconds=12.0,
                        motion_map="affine"):
    """rollout -> frames -> mp4 (audio muxed).

    `renderer` is an IMT ctx dict (load_imt_renderer) or a distilled
    DistilledRenderer module (load_renderer); the IMT path consumes the raw
    motion tokens directly (no range mapping - it is the renderer these
    tokens were born for), the distilled path applies motion_map.
    Shared by this CLI and by the periodic test inside train_emotion.py.
    Returns (raw_motion, stats_string, n_frames).
    """
    motion = rollout_motion(model, helium, cfg, emotion_id, device,
                            gain=gain, max_seconds=max_seconds)
    prefix = os.path.splitext(out_path)[0]
    if isinstance(renderer, dict):
        frames = render_frames_imt(renderer, motion, device, out_prefix=prefix)
        stats = "imt: raw motion tokens, reference-image conditioned (no remap)"
    else:
        mapped, stats = map_motion_for_renderer(motion, renderer, motion_map)
        frames = render_video_frames(renderer, mapped, device, out_prefix=prefix)
    write_video(frames, wav_path, out_path)
    return motion, stats, frames.shape[0]


# ---------------------------------------------------------------------------
# staged white-screen diagnostic for the distilled renderer
# ---------------------------------------------------------------------------
def run_diag(R, model, cfg, h, motion_dir, clip_stem, out_dir, device,
             gain=1.2):
    import torch
    os.makedirs(out_dir, exist_ok=True)

    def probe(tag, motion, n_show=3):
        frames = render_video_frames(
            R, motion, device, out_prefix=os.path.join(out_dir, "diag_" + tag))
        br = frames.reshape(frames.shape[0], -1).mean(1)
        print("[diag] %-22s brightness: %s"
              % (tag, " ".join("%.1f" % b for b in br[:n_show])), flush=True)

    rmean, rstd = R.motion_mean.cpu(), R.motion_std.cpu()
    bf = R.base_frame.cpu()
    print("[diag] base_frame %s mean %.3f min %.3f max %.3f "
          "(a real face is mid-valued, ~0.3-0.6 mean)"
          % (tuple(bf.shape), float(bf.mean()), float(bf.min()),
             float(bf.max())), flush=True)
    print("[diag] motion stats: |mean| %.4f  std %.4f"
          % (float(rmean.abs().mean()), float(rstd.mean())), flush=True)

    print("[diag] stage 1: the renderer's OWN mean motion "
          "-> must reproduce the base face", flush=True)
    probe("1_mean_motion", rmean.unsqueeze(0))

    print("[diag] stage 2: mean +/- k*std on 4 dims (saturation boundary)",
          flush=True)
    for k in (1.0, 2.0, 4.0, 8.0):
        m = rmean.unsqueeze(0).clone()
        m[0, :4] += k * rstd[:4]
        probe("2_plus_%gs" % k, m)
        m = rmean.unsqueeze(0).clone()
        m[0, :4] -= k * rstd[:4]
        probe("2_minus_%gs" % k, m)

    if motion_dir:
        mp = os.path.join(motion_dir, clip_stem + ".pt")
        if os.path.exists(mp):
            m = torch.load(mp, map_location="cpu", weights_only=False)
            if isinstance(m, dict):
                m = m.get("motion", m.get("motion_latents"))
            gt = m.float()[:50]
            print("[diag] stage 3: dataset GT motion raw [%.3f, %.3f] "
                  "sigma [%.1f, %.1f]"
                  % (float(gt.min()), float(gt.max()),
                     float(((gt - rmean) / rstd.clamp_min(1e-6)).min()),
                     float(((gt - rmean) / rstd.clamp_min(1e-6)).max())),
                  flush=True)
            probe("3_gt_raw", gt)
            probe("3_gt_affine",
                  map_motion_for_renderer(gt.clone(), R, "affine")[0])

    print("[diag] stage 4: student rollout motion (neutral + happy)", flush=True)
    for eid in (0, 1):
        mot = rollout_motion(model, h, cfg, eid, device, gain, max_seconds=4.0)
        print("[diag]   rollout[eid=%d] raw [%+.3f, %+.3f] sigma [%+.1f, %+.1f] "
              "NaN=%d" % (eid, float(mot.min()), float(mot.max()),
                          float(((mot - rmean) / rstd.clamp_min(1e-6)).min()),
                          float(((mot - rmean) / rstd.clamp_min(1e-6)).max()),
                          int(torch.isnan(mot).sum())), flush=True)
        probe("4_rollout%d_raw" % eid, mot[:50])
        probe("4_rollout%d_affine" % eid,
              map_motion_for_renderer(mot[:50].clone(), R, "affine")[0])
    print("DIAG DONE -> %s (inspect the diag_*.png frames)" % out_dir, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--base_student", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--helium_dir", required=True)
    ap.add_argument("--raw_audio_dir", required=True)
    ap.add_argument("--clip_stem", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--renderer", default="imt", choices=["imt", "distilled"])
    ap.add_argument("--imt_ckpt", default="",
                    help="old IMTalker renderer.ckpt (required for --renderer imt)")
    ap.add_argument("--ref_image", default="",
                    help="reference face image for the IMT renderer")
    ap.add_argument("--diag", action="store_true",
                    help="run the staged distilled-renderer investigation "
                         "instead of the normal render")
    ap.add_argument("--motion_dir", default="",
                    help="dataset motion dir -> also renders a ground-truth "
                         "motion video with the selected renderer")
    ap.add_argument("--gain", type=float, default=1.2,
                    help="same motion gain the live server applies")
    ap.add_argument("--max_seconds", type=float, default=12.0)
    ap.add_argument("--motion_map", default="affine",
                    choices=["affine", "clamp", "none"],
                    help="distilled renderer only")
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

    hpack = torch.load(os.path.join(args.helium_dir, args.clip_stem + ".pt"),
                       map_location="cpu", weights_only=False)
    h = (hpack["helium"] if isinstance(hpack, dict) else hpack).float()
    T = min(h.shape[0] - cfg.lookahead, int(args.max_seconds * 12.5))
    print("[render] clip %s: %d steps (%.1f s)"
          % (args.clip_stem, T, T / 12.5), flush=True)
    wav_path = find_audio(args.raw_audio_dir, args.clip_stem)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- diagnostic mode: investigate the distilled renderer, then exit ----
    if args.diag:
        R, rpack = load_renderer(repo, device)
        run_diag(R, model, cfg, h, args.motion_dir or "", args.clip_stem,
                 args.out_dir, device, gain=args.gain)
        return


    # ---- build the selected renderer ----
    if args.renderer == "imt":
        assert args.imt_ckpt and os.path.exists(args.imt_ckpt), \
            "--imt_ckpt missing: %s" % args.imt_ckpt
        assert args.ref_image, "--ref_image required for --renderer imt"
        renderer = load_imt_renderer(args.imt_ckpt, args.ref_image, device)
    else:
        renderer, rpack = load_renderer(repo, device)
        print("[render] distilled renderer loaded (val_PSNR=%.2f dB)"
              % rpack.get("val_psnr", 0.0), flush=True)
        rmean, rstd = renderer.motion_mean.cpu(), renderer.motion_std.cpu()
        smean, sstd = model.motion_mean.cpu(), model.motion_std.cpu()
        print("[render] student  motion stats: mean |%.3f|  std %.3f"
              % (float(smean.abs().mean()), float(sstd.mean())), flush=True)
        print("[render] renderer motion stats: mean |%.3f|  std %.3f"
              % (float(rmean.abs().mean()), float(rstd.mean())), flush=True)

    # ---- ground-truth motion through the selected renderer ----
    if args.motion_dir:
        mp = os.path.join(args.motion_dir, args.clip_stem + ".pt")
        if os.path.exists(mp):
            m = torch.load(mp, map_location="cpu", weights_only=False)
            if isinstance(m, dict):
                m = m.get("motion", m.get("motion_latents"))
            gt = m.float()[: 2 * T]
            if isinstance(renderer, dict):
                frames = render_frames_imt(
                    renderer, gt, device,
                    out_prefix=os.path.join(args.out_dir, "gt_imt"))
                out = os.path.join(args.out_dir, "gt_imt_%s.mp4" % args.clip_stem)
                write_video(frames, wav_path, out)
                print("[render] GT (imt) -> %s" % out, flush=True)
            else:
                for mode in ("none", "affine"):
                    m2, _st = map_motion_for_renderer(gt.clone(), renderer, mode)
                    frames = render_video_frames(
                        renderer, m2, device,
                        out_prefix=os.path.join(args.out_dir, "gt_%s" % mode))
                    out = os.path.join(args.out_dir, "gt_%s_%s.mp4"
                                       % (mode, args.clip_stem))
                    write_video(frames, wav_path, out)
                    print("[render] GT %s -> %s" % (mode, out), flush=True)

    # ---- one video per emotion ----
    motions = {}
    for eid in range(NUM_EMOTIONS):
        out_path = os.path.join(args.out_dir, "%d_%s.mp4" % (eid, EMOTION_NAMES[eid]))
        motion, stats, nfr = render_motion_video(
            model, renderer, h, cfg, eid, wav_path, out_path, device,
            gain=args.gain, max_seconds=args.max_seconds,
            motion_map=args.motion_map)
        motions[eid] = motion
        print("[render] %-8s -> %s (%d frames)"
              % (EMOTION_NAMES[eid], out_path, nfr), flush=True)
        print("[render]   %s" % stats, flush=True)

    ref = motions[0]
    print("[render] mean |motion - neutral| per emotion (raw units):", flush=True)
    for eid in range(1, NUM_EMOTIONS):
        d = (motions[eid] - ref).abs().mean().item()
        print("[render]   %-9s %.4f" % (EMOTION_NAMES[eid], d), flush=True)
    print("RENDER DONE -> %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()

