"""Train the emotion conditioner + LoRA adapters on the Helium motion student.

Plain PyTorch (no Lightning in this env). Objective: teacher-forced MSE on
normalized motion over sliding windows, with the cfg.lookahead label shift the
student was distilled with, and 10% classifier-free emotion dropout - the same
training signal shape as the old FM pipeline's emotion stage.
"""
import argparse, glob, json, os, random, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--helium_dir", required=True)
    ap.add_argument("--motion_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_student", required=True)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=float, default=64.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cond_lr_mult", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--window_steps", type=int, default=64)
    ap.add_argument("--emotion_dropout", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--val_split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    # periodic test videos (the old pipeline's every-10-epochs test, adapted to
    # the new architecture: the ref image is baked into the distilled renderer,
    # and the audio features are the PRE-EXTRACTED helium of a fixed clip - the
    # 7B trunk never runs inside the training process)
    ap.add_argument("--test_every_n_epochs", type=int, default=10, help="0 = off")
    ap.add_argument("--test_helium", default="",
                    help="pre-extracted helium .pt for the test clip")
    ap.add_argument("--test_wav", default="", help="audio muxed into the test videos")
    ap.add_argument("--test_out_dir", default="")
    ap.add_argument("--test_emotion_id", type=int, default=-1,
                    help="-1 = cycle all six emotions across tests")
    ap.add_argument("--test_gain", type=float, default=1.2)
    ap.add_argument("--test_seconds", type=float, default=6.0)
    ap.add_argument("--motion_map", default="affine", choices=["affine", "clamp", "none"])
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    src = repo / "src"
    os.environ.setdefault("IMTALKER_DIR", str(src / "imtalker"))
    os.environ.setdefault("DESIGN_A_DIR", str(src / "niloys"))
    for p in (str(src / "imtalker"), str(src / "niloys"),
              str(src / "niloys" / "emotion")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from emotion_student import (EMOTION_NAMES, NUM_EMOTIONS, apply_emotion_lora,
                                 load_base_student, parse_emotion_from_stem,
                                 save_emotion_adapter)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"

    model, cfg, pack = load_base_student(args.base_student)
    model.emotion_conditioner.dropout = args.emotion_dropout
    n_lora = apply_emotion_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                                dropout=args.lora_dropout)
    mot_mean = model.motion_mean.detach().cpu().clone()
    mot_std = model.motion_std.detach().cpu().clone()
    model.to(device)
    print("[train] base student: %s" % args.base_student, flush=True)
    print("[train]   lookahead=%d attn_window=%d d_model=%d n_layers=%d"
          % (cfg.lookahead, cfg.attn_window, cfg.d_model, cfg.n_layers), flush=True)
    print("[train] LoRA wrapped %d layers (rank=%d alpha=%.0f dropout=%.2f)"
          % (n_lora, args.lora_rank, args.lora_alpha, args.lora_dropout), flush=True)

    L, W = cfg.lookahead, args.window_steps

    # ---------------- dataset ----------------
    def _load_clip(stem):
        hp = os.path.join(args.helium_dir, stem + ".pt")
        mp = os.path.join(args.motion_dir, stem + ".pt")
        hpack = torch.load(hp, map_location="cpu", weights_only=False)
        h = hpack["helium"] if isinstance(hpack, dict) else hpack        # [T,4096]
        m = torch.load(mp, map_location="cpu", weights_only=False)
        if isinstance(m, dict):
            m = m.get("motion", m.get("motion_latents"))
        return h.float(), m.float()

    class WindowDataset(Dataset):
        """Random windows: input helium[a+L : a+L+W] -> target motion[2a : 2(a+W)].

        The lookahead shift L matches how the student was distilled: the output
        for step s pairs with helium[s+L] (video lags audio by L*80 ms live).
        prev_motion for window step j is motion[2(a+j)-2 : 2(a+j)] - the two
        frames preceding the target pair, exactly what the streaming state holds.
        """
        def __init__(self, stems):
            self.stems = list(stems)

        def __len__(self):
            return len(self.stems)

        def __getitem__(self, i):
            stem = self.stems[i]
            eid = parse_emotion_from_stem(stem)
            h, m = _load_clip(stem)
            T = min(h.shape[0], m.shape[0] // 2)
            lo, hi = args.warmup_steps, T - W - L
            a = random.randint(lo, max(lo, hi))
            inp = h[a + L: a + L + W]                                   # [W,4096] raw
            tgt = m[2 * a: 2 * (a + W)].view(W, 2, -1)                  # [W,2,32] raw
            if a > 0:
                prev = m[2 * a - 2: 2 * a + 2 * (W - 1)]
            else:
                prev = torch.cat([torch.zeros(2, m.shape[1]), m[: 2 * (W - 1)]], 0)
            tgt_n = (tgt - mot_mean) / mot_std
            prev_n = ((prev.view(W, 2, -1) - mot_mean) / mot_std).reshape(W, -1)
            return inp, prev_n.float(), tgt_n.float(), eid

    # stratified split by emotion, val capped at 200 clips
    by_emo = {e: [] for e in range(NUM_EMOTIONS)}
    for f in sorted(glob.glob(os.path.join(args.helium_dir, "*.pt"))):
        stem = Path(f).stem
        e = parse_emotion_from_stem(stem)
        if e is None:
            continue
        h, m = _load_clip(stem)
        if min(h.shape[0], m.shape[0] // 2) - W - L > args.warmup_steps:
            by_emo[e].append(stem)
    rng = random.Random(args.seed)
    train_stems, val_stems = [], []
    for e in range(NUM_EMOTIONS):
        ss = by_emo[e]
        rng.shuffle(ss)
        n_val = max(1, int(len(ss) * args.val_split)) if len(ss) > 4 else 1
        val_stems += ss[:n_val]
        train_stems += ss[n_val:]
    val_stems = sorted(val_stems)[:200]
    train_ds, val_ds = WindowDataset(train_stems), WindowDataset(val_stems)
    print("[train] clips: %d train / %d val (per emotion: %s)"
          % (len(train_ds), len(val_ds),
             {EMOTION_NAMES[e]: len(by_emo[e]) for e in range(NUM_EMOTIONS)}),
          flush=True)
    assert len(train_ds) > 0, "no usable training clips - check the extraction step"
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True, pin_memory=True,
                        persistent_workers=True)

    # ---------------- optimiser (mirrors the old pipeline's two lr groups) ----------------
    cond_params = [p for n, p in model.named_parameters()
                   if "emotion_conditioner" in n and p.requires_grad]
    lora_params = [p for n, p in model.named_parameters()
                   if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
    print("[train] trainable params: %d (conditioner %d + LoRA %d)"
          % (sum(p.numel() for p in cond_params + lora_params),
             sum(p.numel() for p in cond_params),
             sum(p.numel() for p in lora_params)), flush=True)
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr},
         {"params": cond_params, "lr": args.lr * args.cond_lr_mult}],
        weight_decay=args.weight_decay)

    def _meta(step, val=None):
        return {"step": step, "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "base_student": os.path.basename(args.base_student),
                "student_cfg": pack["cfg"], "emotion_names": EMOTION_NAMES,
                "val": val or {}, "args": vars(args)}

    @torch.no_grad()
    def evaluate():
        model.eval()
        per_emo = {e: [] for e in range(NUM_EMOTIONS)}
        for i in range(len(val_ds)):
            inp, prev, tgt, eid = val_ds[i]
            pred, _a, _r, _g = model.forward_with_emotion(
                inp[None].to(device), prev[None].to(device),
                torch.tensor([eid], device=device), train=False)
            per_emo[eid].append(
                F.mse_loss(pred.float(), tgt[None].to(device).float()).item())
        model.train()
        return {EMOTION_NAMES[e]: (sum(v) / len(v) if v else float("nan"))
                for e, v in per_emo.items()}

    @torch.no_grad()
    def rollout_nmse():
        """Free-running NMSE on one val clip per emotion - the number that
        predicts live behaviour (the student's own tf_nmse metric style)."""
        model.eval()
        out = {}
        for e in range(NUM_EMOTIONS):
            stems_e = [s for s in val_ds.stems if parse_emotion_from_stem(s) == e]
            if not stems_e:
                continue
            h, m = _load_clip(stems_e[0])
            T = min(h.shape[0], m.shape[0] // 2) - L
            st = model.init_state(1, device, torch.float32)
            model.set_emotion(e)
            outs = []
            for t in range(T):
                # step() -> [1,2,32]: BOTH motion frames of the 80 ms step (the
                # state's prev_motion auto-feeds the free-running rollout)
                outs.append(model.step(h[t + L: t + L + 1].to(device), st)[0])
            pred = torch.cat(outs, 0).float().cpu()                 # [2T,32] norm
            tgt = (m[: 2 * T] - mot_mean) / mot_std
            out[EMOTION_NAMES[e]] = float(
                ((pred - tgt) ** 2).mean() / (tgt ** 2).mean().clamp_min(1e-8))
        model.train()
        return out

    @torch.no_grad()
    def emotion_vectors():
        model.eval()
        # stack (not cat): each bias_for(...)[0, 0] is [d_model]; we want [NUM_EMOTIONS, d_model]
        b = torch.stack([model.emotion_conditioner.bias_for(e, device, torch.float32)[0, 0]
                         for e in range(NUM_EMOTIONS)], 0)
        model.train()
        norms = b.norm(dim=-1)
        cos = F.cosine_similarity(b[None, :, :], b[:, None, :], dim=-1)
        return norms, cos

    # ---------------- periodic test videos ----------------
    # Adapted from the old FM pipeline's EpochTestVideoCallback (every 10 epochs,
    # ref image + audio + emotion -> video). New-architecture translation:
    #   * ref image   -> the distilled renderer's baked-in identity (no ref needed)
    #   * audio feats -> the PRE-EXTRACTED helium of a fixed test clip (the 7B
    #                    trunk never runs inside the training process)
    #   * Lightning epoch callback -> an epoch boundary in this step-based loop
    # Rendering is no_grad + eval/train bracketed and reuses one lazily-loaded
    # renderer; it cannot touch the optimizer or the emotion bias
    # (forward_with_emotion saves/restores it; set_emotion writes the live buffer).
    steps_per_epoch = max(1, -(-len(train_ds) // args.batch_size))
    test_every = (args.test_every_n_epochs * steps_per_epoch
                  if (args.test_every_n_epochs > 0 and args.test_helium) else 0)
    test_state = {"R": None, "h": None, "n": 0}
    if test_every > 0:
        import render_emotion_tests as ret
        hpack = torch.load(args.test_helium, map_location="cpu", weights_only=False)
        test_state["h"] = (hpack["helium"] if isinstance(hpack, dict) else hpack).float()
        print("[test] periodic test videos: every %d epochs (%d steps), %.0f s clip -> %s"
              % (args.test_every_n_epochs, test_every, args.test_seconds,
                 args.test_out_dir), flush=True)

    @torch.no_grad()
    def run_test_video(epoch):
        if test_state["h"] is None:
            return
        if test_state["R"] is None:
            test_state["R"], _rp = ret.load_renderer(args.repo_root, device)
        eid = (test_state["n"] % NUM_EMOTIONS
               if args.test_emotion_id < 0 else args.test_emotion_id)
        os.makedirs(args.test_out_dir, exist_ok=True)
        out = os.path.join(args.test_out_dir, "epoch_%04d_%s.mp4"
                           % (epoch, EMOTION_NAMES[eid]))
        model.eval()
        ret.render_motion_video(model, test_state["R"], test_state["h"], cfg, eid,
                                args.test_wav or None, out, device,
                                gain=args.test_gain, max_seconds=args.test_seconds,
                                motion_map=args.motion_map)
        model.train()
        test_state["n"] += 1
        print("[test] epoch %d -> %s" % (epoch, out), flush=True)

    run_test_video(0)   # baseline: the untrained adapter on the test clip

    # ---------------- loop ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    step, ema, t0 = 0, None, time.time()
    it = iter(loader)
    while step < args.max_steps:
        try:
            inp, prev, tgt, eid = next(it)
        except StopIteration:
            it = iter(loader)
            inp, prev, tgt, eid = next(it)
        inp = inp.to(device, non_blocking=True)
        prev = prev.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        eid = eid.to(device, non_blocking=True)
        pred, _a, _r, _g = model.forward_with_emotion(inp, prev, eid, train=True)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(cond_params + lora_params, args.grad_clip)
        opt.step()
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 50 == 0:
            print("[train] step %6d/%d  loss %.5f  ema %.5f  |g| %.3f  %.1f it/s"
                  % (step, args.max_steps, loss.item(), ema, float(gn),
                     step / (time.time() - t0)), flush=True)
        if test_every > 0 and (step % test_every == 0 or step == args.max_steps):
            run_test_video(step // steps_per_epoch)
        if step % args.val_every == 0 or step == args.max_steps:
            mse = evaluate()
            nmse = rollout_nmse()
            norms, cos = emotion_vectors()
            off = cos[~torch.eye(NUM_EMOTIONS, dtype=torch.bool, device=cos.device)]
            print("[val] step %d  teacher-forced MSE per emotion: %s"
                  % (step, {k: round(v, 5) for k, v in mse.items()}), flush=True)
            print("[val]   free-running NMSE: %s"
                  % {k: round(v, 4) for k, v in nmse.items()}, flush=True)
            print("[val]   emotion bias norms: %s  pairwise cos: min %.3f max %.3f"
                  % ([round(float(x), 3) for x in norms],
                     float(off.min()), float(off.max())), flush=True)
            save_emotion_adapter(model, os.path.join(args.out_dir, "last.pt"),
                                 _meta(step, {"tf_mse": mse, "rollout_nmse": nmse}))
        if step % args.save_every == 0:
            save_emotion_adapter(model,
                                 os.path.join(args.out_dir, "step_%06d.pt" % step),
                                 _meta(step))
    print("TRAINING DONE: %d steps, adapter -> %s"
          % (step, os.path.join(args.out_dir, "last.pt")), flush=True)


if __name__ == "__main__":
    main()
