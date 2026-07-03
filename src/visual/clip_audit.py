"""Visual corroboration: do customer photos agree with the listing photos?

Design principle (PROJECT_PLAN §2): images CORROBORATE text evidence, they
never flag a product alone. Studio-vs-phone photos differ in lighting, pose
and background far more than in fabric, so raw similarity thresholds on whole
images would flood false positives. We therefore:

  1. compare CLIP image embeddings (overall appearance agreement), and
  2. compare HSV color histograms (color/sheen proxy),
  3. optionally strip backgrounds first if `rembg` is installed
     (`uv add rembg` — optional, ~200MB extra; falls back to full frame),

CALIBRATION RESULT (2026-07-03, 10 flagged vs 10 control products, two conditions):
    whole frame:    flagged clip=0.648 color=0.689 | control clip=0.659 color=0.598
    rembg crops:    flagged clip=0.727 color=0.711 | control clip=0.738 color=0.686
The distributions OVERLAP in BOTH conditions: segmentation raises similarity
for everyone (backgrounds removed) but does not separate complained-about
products from clean ones. Image-level appearance comparison cannot see texture
at this granularity — the original conv.md plan's "flag if CLIP < 0.75" would
have flagged 20/20 products. Verdicts are therefore conservative: INCONCLUSIVE
unless the scores sit outside everything observed in the control group. The
only remaining visual avenue is multimodal-LLM reasoning over image pairs
(Bedrock — pending account access); text remains the primary signal by design.
"""

import io
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

CLIP_NAME = "openai/clip-vit-base-patch32"
UA = {"User-Agent": "true-texture-research-prototype/0.1"}

# Outlier bounds sit outside the control band of BOTH experiment conditions
# (whole frame: clip 0.55-0.84, color 0.37-0.83; crops: clip 0.62-0.89,
# color 0.33-0.89). Only scores clearly beyond them count as visual support.
OUTLIER_CLIP_BELOW = 0.50
OUTLIER_COLOR_ABOVE = 0.92

try:
    from rembg import remove as _rembg_remove
    HAVE_REMBG = True
except ImportError:
    HAVE_REMBG = False


def first_url(image_entry: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if image_entry.get(k):
            return image_entry[k]
    return None


def official_urls(product: dict, limit: int = 2) -> list[str]:
    urls = []
    for entry in product.get("images", []):
        u = first_url(entry, ("hi_res", "large", "thumb"))
        if u:
            urls.append(u)
    return urls[:limit]


def review_urls(reviews: list[dict], limit: int = 4) -> list[str]:
    urls = []
    for r in reviews:
        for entry in r.get("images", []):
            u = first_url(entry, ("large_image_url", "medium_image_url",
                                  "small_image_url", "attachment_url"))
            if u:
                urls.append(u)
    return urls[:limit]


def fetch_image(url: str, cache_path: Path) -> Image.Image | None:
    try:
        if not cache_path.exists():
            resp = requests.get(url, headers=UA, timeout=20)
            resp.raise_for_status()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(resp.content)
        img = Image.open(io.BytesIO(cache_path.read_bytes())).convert("RGB")
        return img
    except Exception as e:  # network/decode failures shouldn't kill the run
        print(f"    skip image ({e.__class__.__name__}): {url[:80]}")
        return None


def isolate_garment(img: Image.Image) -> Image.Image:
    if not HAVE_REMBG:
        return img
    try:
        cut = _rembg_remove(img)
        rgb = Image.new("RGB", cut.size, (255, 255, 255))
        rgb.paste(cut, mask=cut.split()[-1])
        return rgb
    except Exception:
        return img


def color_histogram(img: Image.Image, bins: int = 24) -> np.ndarray:
    hsv = np.asarray(img.convert("HSV").resize((128, 128)), dtype=np.float32)
    hist, _ = np.histogram(hsv[..., 0], bins=bins, range=(0, 255),
                           weights=hsv[..., 1] / 255.0)  # saturation-weighted hue
    total = hist.sum()
    return hist / total if total > 0 else hist


class ClipAuditor:
    def __init__(self):
        # Vision tower only (no text tower) — returns projected image_embeds
        # via a stable interface across transformers versions.
        self.model = CLIPVisionModelWithProjection.from_pretrained(CLIP_NAME)
        self.processor = CLIPImageProcessor.from_pretrained(CLIP_NAME)
        self.model.eval()

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        feats = self.model(**inputs).image_embeds
        return feats / feats.norm(dim=-1, keepdim=True)

    def audit(self, official: list[Image.Image],
              review: list[Image.Image]) -> dict:
        official = [isolate_garment(i) for i in official]
        review = [isolate_garment(i) for i in review]
        e_off = self.embed(official)
        e_rev = self.embed(review)
        sims = (e_rev @ e_off.T).max(dim=1).values  # best official match per review photo

        h_off = np.mean([color_histogram(i) for i in official], axis=0)
        deltas = [0.5 * np.abs(color_histogram(i) - h_off).sum() for i in review]

        clip_mean = float(sims.mean())
        clip_min = float(sims.min())
        color_mean = float(np.mean(deltas))
        if clip_mean < OUTLIER_CLIP_BELOW or color_mean > OUTLIER_COLOR_ABOVE:
            verdict = "SUPPORTS_OUTLIER"  # outside everything the control group produced
        else:
            verdict = "INCONCLUSIVE"      # v1 whole-frame metric can't separate (see docstring)
        return {
            "clip_similarity_mean": round(clip_mean, 3),
            "clip_similarity_min": round(clip_min, 3),
            "color_delta_mean": round(color_mean, 3),
            "background_removed": HAVE_REMBG,
            "visual_corroboration": verdict,
        }
