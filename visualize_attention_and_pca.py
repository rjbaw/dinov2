import torch, math, numpy as np, warnings
from PIL import Image
from torchvision import transforms
from dinov2.models.vision_transformer import DinoVisionTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="xFormers is available")

model = DinoVisionTransformer(
    img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, block_chunks=4, init_values=1e-5
).eval()
ckpt = torch.load("test.pth", map_location="cpu", weights_only=False)

sd = {k[len("student.backbone.") :]: v for k, v in ckpt.items() if k.startswith("student.backbone.")}
model.load_state_dict(sd, strict=False)

prep = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
img_pil = Image.open("test.jpg").convert("RGB")
img = prep(img_pil).unsqueeze(0)  # (1,3,H_in,W_in)
H_in, W_in = img.shape[-2], img.shape[-1]


def last_attention(root):
    for m in reversed(list(root.modules())):
        if m.__class__.__name__.lower().endswith("attention"):
            return m
    raise RuntimeError("No Attention module found")


attn_mod = last_attention(model)

attn_buf, tok_buf = [], []


def capture_attn(module, inp, out):
    x = inp[0]
    B, N_tok, _ = x.shape
    qkv = module.qkv(x).reshape(B, N_tok, 3, module.num_heads, -1)
    q, k, _ = qkv.permute(2, 0, 3, 1, 4)
    scale = getattr(module, "scale", 1.0 / math.sqrt(k.size(-1)))
    attn = (q @ k.transpose(-2, -1)) * scale
    attn_buf.append(attn.softmax(dim=-1).detach().cpu())  # (B,H,N,N)


def capture_tokens(_module, _inp, out):
    tok_buf.append(out.detach().cpu())  # (B,1+N,C) post-norm


hook_attn = attn_mod.register_forward_hook(capture_attn)
hook_norm = model.norm.register_forward_hook(capture_tokens)

with torch.no_grad():
    _ = model(img)

hook_attn.remove()
hook_norm.remove()

attn_last = attn_buf[-1]  # (1, heads, N, N)
tokens = tok_buf[-1]  # (1, 1+N, C)  **post-norm**
patch = tokens[:, 1:, :]  # drop CLS → (1, N, C)

N = patch.shape[1]
h = int(round(math.sqrt(N)))
while N % h:
    h -= 1
H, W = h, N // h

patch = F.normalize(patch, p=2, dim=-1)  # (1,N,C)  L2-normalize
X = patch.squeeze(0).numpy()  # (N,C)

pca = PCA(n_components=3, svd_solver="full", whiten=False)
Xc = pca.fit_transform(X)  # (N,3)

signs = np.sign(pca.components_[np.arange(3), np.abs(pca.components_).argmax(axis=1)])
Xc *= signs.reshape(1, 3)

rgb = Xc.reshape(H, W, 3)
rgb -= rgb.min((0, 1), keepdims=True)
rgb /= np.ptp(rgb, (0, 1), keepdims=True) + 1e-8

rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()  # (1,3,H,W)
rgb_t = F.interpolate(rgb_t, size=(H_in, W_in), mode="nearest")
rgb_img = (rgb_t.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
Image.fromarray(rgb_img).save("pca_map.jpg")

cls2patch = attn_last.mean(1)[:, 0, 1:]  # (1, N)
h_att = int(round(math.sqrt(cls2patch.shape[-1])))
while cls2patch.shape[-1] % h_att:
    h_att -= 1
H_att, W_att = h_att, cls2patch.shape[-1] // h_att

heat = cls2patch.reshape(1, 1, H_att, W_att)
heat = F.interpolate(heat, size=(H_in, W_in), mode="bilinear", align_corners=False)
heat = heat.squeeze().numpy()
heat = (heat - heat.min()) / (np.ptp(heat) + 1e-8)

plt.imsave("attention_map.jpg", heat, cmap="inferno")
print(f"saved pca_map.jpg and attention_map.jpg (grid {H}×{W})")
