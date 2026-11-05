"""Vendored reference-image-conditioned IMTalker renderer (from the old
pipeline's renderer/ package), used by the emotion pipeline's periodic and
final test videos. It consumes the same 32-D motion tokens the motion student
predicts (the student was distilled from the old FM generator whose sample()
output fed this renderer), so any reference image can drive the identity.

Only the inference path is vendored (models + its 3 dependency modules);
training/discriminator/VGG code is intentionally left out. Two import shims
were applied: the timm helpers in attention_modules.py and the unused
matplotlib/torchvision/PIL imports in modules.py.
"""
