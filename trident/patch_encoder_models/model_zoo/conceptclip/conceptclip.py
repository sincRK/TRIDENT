"""ConceptCLIP inference helpers.

Vendored open_clip fork from https://github.com/JerrryNie/ConceptCLIP
(pre_training/src/open_clip). Weights are loaded from Hugging Face.
"""
import os
import sys
import types
from typing import Optional

from torchvision import transforms
from torchvision.transforms import InterpolationMode

_CONCEPTCLIP_ROOT = os.path.dirname(os.path.abspath(__file__))


def ensure_open_clip_path() -> None:
    """Expose vendored open_clip for HF trust_remote_code imports."""
    if _CONCEPTCLIP_ROOT not in sys.path:
        sys.path.insert(0, _CONCEPTCLIP_ROOT)


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def patch_model_forward(model):
    original_forward = model.forward

    def patched_forward(self, image=None, text=None, **kwargs):
        return original_forward(pixel_values=image, input_ids=text, **kwargs)

    model.forward = types.MethodType(patched_forward, model)
    return model


def load_model_from_pretrained(
    weights_path: Optional[str] = None,
    hf_repo_id: str = "JerrryNie/ConceptCLIP",
):
    ensure_open_clip_path()
    from transformers import AutoModel

    if weights_path:
        model_dir = os.path.dirname(weights_path)
        model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        )
    else:
        model = AutoModel.from_pretrained(hf_repo_id, trust_remote_code=True)

    return patch_model_forward(model)
