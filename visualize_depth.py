import math
import itertools
import os
from functools import partial

import torch
import torch.nn.functional as F

from dinov2.eval.depth.models import build_depther


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther


BACKBONE_SIZE = "small"  # in ("small", "base", "large" or "giant")

backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load(repo_or_dir=".", model=backbone_name, source="local", pretrained=False)
backbone_model.eval()
backbone_model.cuda()

import urllib

from mmengine.config import Config
from mmengine.runner import load_checkpoint


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


HEAD_DATASET = "nyu"  # in ("nyu", "kitti")
HEAD_TYPE = "dpt"  # in ("linear", "linear4", "dpt")


DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

# Prefer local config if present to avoid network in restricted envs.
local_cfg_filename = f"{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
cfg = None
if os.path.exists(local_cfg_filename):
    with open(local_cfg_filename, "r", encoding="utf-8") as f:
        cfg_str = f.read()
    cfg = Config.fromstring(cfg_str, file_format=".py")
else:
    try:
        cfg_str = load_config_from_url(head_config_url)
        cfg = Config.fromstring(cfg_str, file_format=".py")
    except Exception:
        embed_dims = {"small": 384, "base": 768, "large": 1024, "giant": 1536}[BACKBONE_SIZE]
        cfg = Config(
            dict(
                model=dict(
                    type="DepthEncoderDecoder",
                    backbone=dict(
                        type="DinoVisionTransformer", out_indices=[3], output_cls_token=True, final_norm=False
                    ),
                    decode_head=dict(
                        type="DPTHead",
                        in_channels=[embed_dims, embed_dims, embed_dims, embed_dims],
                        channels=256,
                        min_depth=1e-3,
                        max_depth=10.0,
                        classify=False,
                        n_bins=256,
                        bins_strategy="UD",
                        norm_strategy="linear",
                        scale_up=False,
                        align_corners=False,
                    ),
                    train_cfg=dict(),
                    test_cfg=dict(mode="whole", stride=(128, 128), crop_size=(640, 640)),
                )
            )
        )

model = create_depther(
    cfg,
    backbone_model=backbone_model,
    backbone_size=BACKBONE_SIZE,
    head_type=HEAD_TYPE,
)

try:
    local_head_ckpt = f"{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"
    if os.path.exists(local_head_ckpt):
        load_checkpoint(model, local_head_ckpt, map_location="cpu")
    else:
        load_checkpoint(model, head_checkpoint_url, map_location="cpu")
except Exception:
    pass
model.eval()
model.cuda()

import urllib

from PIL import Image


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
local_image = "test.jpg"
try:
    if os.path.exists(local_image):
        image = Image.open(local_image).convert("RGB")
    else:
        image = load_image_from_url(EXAMPLE_IMAGE_URL)
except Exception:
    raise RuntimeError("No input image available: place a test image at 'test.jpg' or allow network access.")

import matplotlib
from torchvision import transforms


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3],
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ]
    )


def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True)  # ((1)xhxwx4)
    colors = colors[:, :, :3]  # Discard alpha component
    return Image.fromarray(colors)


transform = make_depth_transform()

scale_factor = 1
rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
transformed_image = transform(rescaled_image)
batch = transformed_image.unsqueeze(0).cuda()

with torch.inference_mode():
    result = model.whole_inference(batch, img_meta=None, rescale=True)

depth_image = render_depth(result.squeeze().cpu())

depth_image.save("depth.jpg")
