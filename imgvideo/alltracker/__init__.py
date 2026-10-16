"""AllTracker (Adam W. Harley et al., https://github.com/aharley/alltracker), vendored: the model only.

MIT licence (LICENSE here), code and released weights (alltracker.pth on Hugging Face, aharley/alltracker). Source
commit e7553135e7b361590dbccd10e2b274b024f41cd6: nets/alltracker.py, nets/blocks.py and utils/misc.py, unchanged except in alltracker.py:
- imports made relative to this package (`from . import misc`, `from .blocks import ...`);
- the two `.cuda()` calls at inference move tensors to the model's own device instead (CPU, CUDA).
Build the model with init_weights=False: the default fetches torchvision's ImageNet ConvNeXt weights over the network,
and the checkpoint replaces every weight anyway. The weights file is not vendored (imgvideo.track.alltracker_weights).
"""
from .alltracker import Net  # noqa: F401
