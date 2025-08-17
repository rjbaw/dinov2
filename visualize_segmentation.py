import math
import itertools
import os
from functools import partial

import torch
import torch.nn.functional as F
from mmseg.apis import init_model, inference_model

import dinov2.eval.segmentation.models

import urllib

import mmcv
import mmengine
from mmengine.runner import load_checkpoint
from mmseg.models.data_preprocessor import SegDataPreProcessor

from PIL import Image

import numpy as np

import dinov2.eval.segmentation.utils.colormaps as colormaps

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

def render_segmentation(segmentation_logits, dataset):
    """Render a segmentation map to an RGB image.

    Notes:
    - `segmentation_logits` is expected to contain class ids in the range
      [0, num_classes-1] for the chosen dataset. Some pipelines may produce
      negative values for ignore regions; we clamp these to background color.
    - Avoid off-by-one indexing: do NOT shift labels.
    """
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    labels = np.asarray(segmentation_logits, dtype=np.int64)
    labels = np.clip(labels, 0, len(colormap_array) - 1)
    segmentation_values = colormap_array[labels]
    return Image.fromarray(segmentation_values)

try:
    from torch.serialization import add_safe_globals as _add_safe_globals
    try:
        from numpy.core.multiarray import scalar as _np_scalar
    except Exception:
        from numpy._core.multiarray import scalar as _np_scalar  # type: ignore[attr-defined]
    import numpy as _np
    _add_safe_globals([_np_scalar, _np.dtype])
except Exception:
    pass

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


def create_segmenter(cfg, backbone_model, device):
    model = init_model(cfg, device=device)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model

BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")

backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

device = "cuda" if torch.cuda.is_available() else "cpu"

backbone_model = torch.hub.load(repo_or_dir=".", model=backbone_name, source="local", pretrained=False)
backbone_model.eval()
backbone_model.to(device)


HEAD_SCALE_COUNT = 3 # more scales: slower but better results, in (1,2,3,4,5)
HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
HEAD_TYPE = "ms" # in ("ms, "linear")


# DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
# head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
# head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"
test_img_path = "test.jpg"

local_cfg_filename = f"{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
if os.path.exists(local_cfg_filename):
    with open(local_cfg_filename, "r", encoding="utf-8") as f:
        cfg_str = f.read()
else:
    try:
        cfg_str = load_config_from_url(head_config_url)
    except Exception as e:
        sample_local_cfg = "dinov2_vits14_voc2012_ms_config.py"
        if os.path.exists(sample_local_cfg):
            with open(sample_local_cfg, "r", encoding="utf-8") as f:
                cfg_str = f.read()
        else:
            raise RuntimeError(f"Unable to load config from URL or local file: {e}")

cfg = mmengine.config.Config.fromstring(cfg_str, file_format=".py")

# def migrate_msfa_and_pack(pipeline, base_scale=None, ratios=None, allow_flip=None):
#     for i, t in enumerate(pipeline):
#         if t.get('type') == 'MultiScaleFlipAug':
#             # Remove legacy MS keys
#             t.pop('img_scale', None)
#             img_ratios = t.pop('img_ratios', None)
#             old_flip = t.pop('flip', None)

#             # Multi-scale factors and flip
#             ratios = ratios or img_ratios or [1.0]
#             allow = allow_flip if allow_flip is not None else (old_flip if old_flip is not None else False)

#             # New MSFA API (MMCV 2.x)
#             t['scale_factor'] = ratios
#             t['allow_flip'] = allow
#             t['resize_cfg'] = dict(type='Resize', keep_ratio=True)

#             # Remove inner Resize (scale is injected via resize_cfg)
#             t['transforms'] = [st for st in t['transforms'] if st.get('type') != 'Resize']

#             # Ensure RandomFlip has prob
#             for st in t['transforms']:
#                 if st.get('type') == 'RandomFlip' and 'prob' not in st:
#                     st['prob'] = 0.0  # deterministic inference; set 0.5 if you want TTA flips

#             # Drop legacy formatting steps and end with PackSegInputs
#             new_transforms = []
#             for st in t['transforms']:
#                 if st.get('type') in ('Collect', 'ImageToTensor', 'DefaultFormatBundle'):
#                     continue
#                 new_transforms.append(st)

#             # Append PackSegInputs if not present
#             if not any(st.get('type') == 'PackSegInputs' for st in new_transforms):
#                 new_transforms.append(dict(type='PackSegInputs'))

#             t['transforms'] = new_transforms

def replace_loader_with_ndarray_loader(pipeline):
    for t in pipeline:
        if t.get('type') == 'LoadImageFromFile':
            t['type'] = 'LoadImageFromNDArray'

def migrate_ms_aug_and_pack(pipeline, head_scale_count=None):
    new_pipeline = []
    for t in pipeline:
        if t.get('type') in ('MultiScaleFlipAug', 'TestTimeAug'):
            base_scale = t.get('img_scale')
            img_ratios = t.get('img_ratios') or [1.0]
            if head_scale_count is not None:
                img_ratios = img_ratios[:head_scale_count]
            r0 = img_ratios[0]
            resize_scale = None
            if isinstance(base_scale, (list, tuple)) and len(base_scale) == 2:
                w0, h0 = base_scale
                resize_scale = (int(w0 * r0), int(h0 * r0))

            inner = []
            for st in t['transforms']:
                if st.get('type') in ('Collect', 'ImageToTensor', 'DefaultFormatBundle'):
                    continue
                if st.get('type') == 'RandomFlip':
                    st = dict(**st)
                    st['prob'] = 0.0
                if st.get('type') == 'Resize':
                    st = dict(**st)
                    if 'scale' not in st and 'scale_factor' not in st and resize_scale is not None:
                        st['scale'] = resize_scale
                inner.append(st)

            if not any(st.get('type') in ('PackSegInputs', 'mmseg.PackSegInputs') for st in inner):
                inner.append(dict(type='mmseg.PackSegInputs'))

            new_pipeline.extend(inner)
        else:
            new_pipeline.append(t)

    pipeline.clear()
    pipeline.extend(new_pipeline)

if hasattr(cfg, 'test_pipeline'):
    replace_loader_with_ndarray_loader(cfg.test_pipeline)
    migrate_ms_aug_and_pack(cfg.test_pipeline, head_scale_count=HEAD_SCALE_COUNT)
if hasattr(cfg, 'data') and hasattr(cfg.data, 'test'):
    replace_loader_with_ndarray_loader(cfg.data.test.pipeline)
    migrate_ms_aug_and_pack(cfg.data.test.pipeline, head_scale_count=HEAD_SCALE_COUNT)
if hasattr(cfg.data, 'val'):
    replace_loader_with_ndarray_loader(cfg.data.val.pipeline)
    migrate_ms_aug_and_pack(cfg.data.val.pipeline, head_scale_count=HEAD_SCALE_COUNT)

# # Apply to all relevant pipelines
# migrate_msfa_and_pack(cfg.data.test.pipeline)
# if hasattr(cfg, 'test_pipeline'):
#     migrate_msfa_and_pack(cfg.test_pipeline)
# if hasattr(cfg.data, 'val'):
#     migrate_msfa_and_pack(cfg.data.val.pipeline)


# # If limiting TTA scales
# if HEAD_TYPE == "ms":
#     if hasattr(cfg, 'test_pipeline') and len(cfg.test_pipeline) > 1 and cfg.test_pipeline[1].get('type') == 'MultiScaleFlipAug':
#         cfg.test_pipeline[1]['scale_factor'] = cfg.test_pipeline[1]['scale_factor'][:HEAD_SCALE_COUNT]
#         print("scale factors:", cfg.test_pipeline[1]['scale_factor'])

# if HEAD_TYPE == "ms":
#     cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
#     print("scales:", cfg.data.test.pipeline[1]["img_ratios"])
# print(cfg)

model = create_segmenter(cfg, backbone_model=backbone_model, device=device)
try:
    model.data_preprocessor = SegDataPreProcessor()
    model.data_preprocessor.to(torch.device(device))
except Exception:
    pass

try:
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'test_pipeline'):
        replace_loader_with_ndarray_loader(model.cfg.test_pipeline)
        migrate_ms_aug_and_pack(model.cfg.test_pipeline, head_scale_count=HEAD_SCALE_COUNT)
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'data') and hasattr(model.cfg.data, 'test'):
        replace_loader_with_ndarray_loader(model.cfg.data.test.pipeline)
        migrate_ms_aug_and_pack(model.cfg.data.test.pipeline, head_scale_count=HEAD_SCALE_COUNT)
except Exception:
    pass

local_head_ckpt = None
candidate = f"{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"
if os.path.exists(candidate):
    local_head_ckpt = candidate
elif os.path.exists("test.pth"):
    local_head_ckpt = "test.pth"

if local_head_ckpt is not None:
    try:
        load_checkpoint(model, local_head_ckpt, map_location="cpu")
    except Exception:
        local_head_ckpt = None
else:
    try:
        load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    except Exception:
        pass

model.eval()

DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}

try:
    print("model.data_preprocessor:", type(getattr(model, 'data_preprocessor', None)))
    print("model.cfg.test_pipeline:", getattr(getattr(model, 'cfg', None), 'test_pipeline', None))
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'data') and hasattr(model.cfg.data, 'test'):
        print("model.cfg.data.test.pipeline:", model.cfg.data.test.pipeline)
except Exception:
    pass

if __name__ == "__main__":
    image = Image.open(test_img_path).convert("RGB")
    array = np.array(image)[:, :, ::-1]  # RGB -> BGR
    result = inference_model(model, array)  # SegDataSample
    seg = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.int64)
    segmented_image = render_segmentation(seg, HEAD_DATASET)
    segmented_image.save("segment.jpg")
