"""Emotion conditioning for the Helium motion student (training + live).

Faithful port of the old FM pipeline's emotion mechanism
(IMTalker generator/FM.py::EmotionConditioner + generator/train_lora.py)
re-targeted at the distilled CausalMotionStudent (src/niloys/student/model.py):

  * StudentEmotionConditioner  one-hot(7) -> Linear(7 -> d_model) -> LayerNorm
                               -> SiLU, added to the student's embedded token
                               stream. 10% of training steps get a zeroed bias
                               (classifier-free dropout), like the old pipeline.
  * EmotionMotionStudent       CausalMotionStudent subclass. The live bias
                               lives in a registered buffer updated IN PLACE by
                               set_emotion(), so the captured CUDA graph keeps
                               reading the same storage when the emotion is
                               switched mid-conversation.
  * LoRA helpers               adapters on the transformer blocks (attn.qkv,
                               attn.proj, mlp.0, mlp.3) and the output heads
                               (head_abs, head_res, head_gate). The Helium input
                               path (helium_ln / helium_proj / prev_enc /
                               fuse_*) is never touched - the same lip-sync
                               guarantee as freezing the old audio projections.

Only NEW files live under src/niloys/emotion/ - nothing existing is modified.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

from student.model import CausalMotionStudent, StudentConfig

EMOTION_MAP = {
    "neutral": 0, "calm": 0,
    "happy": 1, "happiness": 1, "joy": 1, "amused": 1,
    "sad": 2, "sadness": 2,
    "angry": 3, "anger": 3, "rage": 3,
    "surprise": 4, "surprised": 4,
    "fear": 5, "fearful": 5, "afraid": 5,
}
EMOTION_NAMES = ["neutral", "happy", "sad", "angry", "surprise", "fear"]
NUM_EMOTIONS = 6
NUM_CLASSES = NUM_EMOTIONS + 1      # one-hot width incl. the classifier-free null slot
ADAPTER_FORMAT = "student_emotion_adapter_v1"


def parse_emotion_from_stem(stem: str):
    """MEAD-style names: M003_happy_level_1_001, W011_sad_3_012, ..."""
    for part in re.split(r"[_\-\.\s]+", stem.lower()):
        if part in EMOTION_MAP:
            return EMOTION_MAP[part]
    return None


class StudentEmotionConditioner(nn.Module):
    """emotion id -> [B,T,d_model] additive bias (the old EmotionConditioner at d_model=512)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = float(dropout)
        self.emotion_embedding = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        with torch.no_grad():
            self.emotion_embedding.weight.copy_(torch.eye(NUM_CLASSES))
        self.emotion_embedding.weight.requires_grad_(False)
        self.emotion_projection = nn.Linear(NUM_CLASSES, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()

    def forward(self, emotion_ids, batch_size: int, seq_len: int, train: bool = False):
        emb = self.emotion_embedding(emotion_ids)           # [B,7] one-hot
        emb = self.emotion_projection(emb)                  # [B,C]
        emb = self.layer_norm(emb)
        emb = self.activation(emb)
        emb = emb.unsqueeze(1).expand(batch_size, seq_len, emb.shape[-1])
        if train and self.dropout > 0:
            mask = torch.rand(batch_size, 1, 1, device=emb.device) < self.dropout
            emb = torch.where(mask, torch.zeros_like(emb), emb)
        return emb

    @torch.no_grad()
    def bias_for(self, idx: int, device, dtype):
        ids = torch.tensor([int(idx)], device=device, dtype=torch.long)
        return self.forward(ids, 1, 1, train=False).to(device=device, dtype=dtype)

class EmotionMotionStudent(CausalMotionStudent):
    """CausalMotionStudent + emotion bias added after _embed()."""

    def __init__(self, cfg: StudentConfig, emotion_dropout: float = 0.1):
        super().__init__(cfg)
        self.emotion_conditioner = StudentEmotionConditioner(cfg.d_model, emotion_dropout)
        self.register_buffer("emotion_bias", torch.zeros(1, 1, cfg.d_model),
                             persistent=False)
        self._emotion_bias = None       # streamed bias (the buffer) while live

    # -- live --------------------------------------------------------------
    @torch.no_grad()
    def set_emotion(self, idx: int):
        emb = self.emotion_conditioner.bias_for(
            idx, self.emotion_bias.device, self.emotion_bias.dtype)
        self.emotion_bias.copy_(emb)    # in place: the captured CUDA graph keeps working
        self._emotion_bias = self.emotion_bias

    # -- shared ------------------------------------------------------------
    def _embed(self, helium, prev_motion):
        x = super()._embed(helium, prev_motion)
        if self._emotion_bias is not None:
            x = x + self._emotion_bias
        return x

    # -- training ----------------------------------------------------------
    def forward_with_emotion(self, helium, prev_motion, emotion_ids, train: bool = True):
        B, T = helium.shape[0], helium.shape[1]
        bias = self.emotion_conditioner(emotion_ids, B, T, train=train)
        prev = self._emotion_bias
        self._emotion_bias = bias
        try:
            return super().forward(helium, prev_motion)
        finally:
            self._emotion_bias = prev


# ---------------------------------------------------------------- LoRA
LORA_TARGETS = ("attn.qkv", "attn.proj", "mlp.0", "mlp.3",
                "head_abs", "head_res", "head_gate")


def apply_emotion_lora(model, rank: int = 32, alpha: float = 64.0, dropout: float = 0.05):
    """Freeze everything, unfreeze the conditioner, wrap LoRA around the blocks'
    attention/MLP projections and the output heads. Returns #wrapped layers."""
    from generator.train_lora import LoRALinear      # needs src/imtalker on sys.path
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.emotion_conditioner.parameters():
        p.requires_grad_(True)
    model.emotion_conditioner.emotion_embedding.weight.requires_grad_(False)
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if not any(name == t or name.endswith("." + t) for t in LORA_TARGETS):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, LoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout))
        n += 1
    return n


def save_emotion_adapter(model, path, meta):
    from generator.train_lora import LoRALinear
    sd = {"format": ADAPTER_FORMAT, "meta": meta,
          "conditioner": {k: v.detach().cpu()
                          for k, v in model.emotion_conditioner.state_dict().items()},
          "lora": {}}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            sd["lora"][name] = {"A": mod.lora_A.weight.detach().cpu(),
                                "B": mod.lora_B.weight.detach().cpu()}
    torch.save(sd, path)
    return sd


def load_base_student(base_ckpt: str):
    """Fresh trainable copy of the base student (weights + stats buffers ride along)."""
    pack = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    cfg = StudentConfig(**pack["cfg"])
    model = EmotionMotionStudent(cfg)
    model.load_state_dict(pack["model"], strict=False)   # conditioner keys are new
    return model, cfg, pack


def build_emotion_student(base_ckpt: str, adapter_ckpt: str, device,
                          emotion_dropout: float = 0.0):
    """Base student + merged emotion LoRA + loaded conditioner (eval, frozen).

    The LoRA deltas are merged straight into the base Linear weights - the live
    path keeps no LoRA wrappers, so the student's CUDA graph is unaffected.
    """
    model, cfg, _pack = load_base_student(base_ckpt)
    ad = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
    assert ad.get("format") == ADAPTER_FORMAT, \
        "not a student emotion adapter: %s" % adapter_ckpt
    meta = ad.get("meta", {})
    scale = float(meta.get("lora_alpha", 64.0)) / float(meta.get("lora_rank", 32))
    for name, d in ad["lora"].items():
        mod = model.get_submodule(name)
        mod.weight.data.add_(scale * (d["B"] @ d["A"]).to(mod.weight.dtype))
    model.emotion_conditioner.load_state_dict(ad["conditioner"])
    model.emotion_conditioner.dropout = float(emotion_dropout)
    model.to(device).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, meta
