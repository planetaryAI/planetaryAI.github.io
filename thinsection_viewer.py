import io, re, zipfile
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="PPL–XPL Thin Section Viewer", layout="wide")

PPL_ZIP = "PPL.zip"
XPL_ZIP = "XPL.zip"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

@dataclass
class Pair:
    key: str
    ppl: str
    xpl: str

def list_images(z):
    return sorted(
        n for n in z.namelist()
        if n.lower().endswith(IMG_EXTS)
        and not n.split("/")[-1].startswith("._")
    )

def extract_key(name: str) -> str:
    base = name.split("/")[-1]
    stem = re.sub(r"\.[^.]+$", "", base)
    stem = re.sub(r"^(ppl|xpl)[-_ ]*", "", stem, flags=re.I)
    nums = re.findall(r"\d+", stem)
    return (nums[-1].lstrip("0") or "0") if nums else stem.lower()

def pair_images(ppl_names, xpl_names) -> List[Pair]:
    ppl_map = {extract_key(n): n for n in ppl_names}
    xpl_map = {extract_key(n): n for n in xpl_names}
    keys = sorted(ppl_map.keys() & xpl_map.keys(), key=lambda x: int(x) if x.isdigit() else x)
    return [Pair(k, ppl_map[k], xpl_map[k]) for k in keys]

def center_crop(a, b):
    H = min(a.shape[0], b.shape[0])
    W = min(a.shape[1], b.shape[1])
    def c(im):
        y = (im.shape[0] - H) // 2
        x = (im.shape[1] - W) // 2
        return im[y:y+H, x:x+W]
    return c(a), c(b)

def gray_norm(im):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)

def edges(g):
    u8 = ((g - g.min()) / (np.ptp(g) + 1e-6) * 255).astype(np.uint8)
    return cv2.Canny(u8, 50, 150).astype(np.float32)

def estimate_shift(ref, mov):
    (dx, dy), _ = cv2.phaseCorrelate(ref, mov)
    return float(dx), float(dy)

def warp(im, dx, dy, out_wh):
    W, H = out_wh
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(
        im, M, (W, H),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT
    )

def resize_max_width(rgb: np.ndarray, max_w: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w <= max_w:
        return rgb
    scale = max_w / w
    new_h = int(h * scale)
    return cv2.resize(rgb, (max_w, new_h), interpolation=cv2.INTER_AREA)

def wipe(left, right, pct):
    W = left.shape[1]
    x = int(W * pct / 100)
    out = left.copy()
    out[:, x:] = right[:, x:]
    out[:, max(0, x-1):min(W, x+1)] = [255, 255, 255]
    return out

# ---- CACHING ----
@st.cache_data(show_spinner=False)
def load_zip_lists(ppl_zip_path: str, xpl_zip_path: str):
    ppl_z = zipfile.ZipFile(ppl_zip_path)
    xpl_z = zipfile.ZipFile(xpl_zip_path)
    ppl_list = list_images(ppl_z)
    xpl_list = list_images(xpl_z)
    pairs = pair_images(ppl_list, xpl_list)
    return ppl_list, xpl_list, pairs

@st.cache_data(show_spinner=False)
def load_pair_images(ppl_zip_path: str, xpl_zip_path: str, ppl_name: str, xpl_name: str):
    with zipfile.ZipFile(ppl_zip_path) as pz:
        ppl = np.array(Image.open(io.BytesIO(pz.read(ppl_name))).convert("RGB"))
    with zipfile.ZipFile(xpl_zip_path) as xz:
        xpl = np.array(Image.open(io.BytesIO(xz.read(xpl_name))).convert("RGB"))
    return ppl, xpl

@st.cache_data(show_spinner=False)
def align_pair(ppl_rgb: np.ndarray, xpl_rgb: np.ndarray):
    ppl, xpl = center_crop(ppl_rgb, xpl_rgb)

    # Cross-modality robust shift using edges
    dx, dy = estimate_shift(edges(gray_norm(xpl)), edges(gray_norm(ppl)))

    H, W = xpl.shape[:2]
    ppl_aligned = warp(ppl, dx, dy, (W, H))
    return ppl_aligned, xpl, dx, dy


st.title("PPL ↔ XPL Thin Section Viewer (Fast)")

with st.sidebar:
    st.header("Display")
    preview_w = st.slider("Preview width (px)", 700, 1800, 1024, 50)
    st.caption("Smaller = faster slider.")
    show_full_side_by_side = st.checkbox("Also show full-res side-by-side (slower)", value=False)

ppl_list, xpl_list, pairs = load_zip_lists(PPL_ZIP, XPL_ZIP)

st.caption(f"Using {len(pairs)} matched image pairs from {PPL_ZIP} and {XPL_ZIP}")

if not pairs:
    st.error("No matching PPL/XPL pairs found. Check filenames.")
    st.stop()

if "i" not in st.session_state:
    st.session_state.i = 0

c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    if st.button("⬅ Prev"):
        st.session_state.i = max(0, st.session_state.i - 1)
with c3:
    if st.button("Next ➡"):
        st.session_state.i = min(len(pairs) - 1, st.session_state.i + 1)
with c2:
    st.session_state.i = st.slider("Image pair", 0, len(pairs) - 1, st.session_state.i)

pair = pairs[st.session_state.i]

# load + align (cached)
ppl_raw, xpl_raw = load_pair_images(PPL_ZIP, XPL_ZIP, pair.ppl, pair.xpl)
ppl_aligned, xpl, dx, dy = align_pair(ppl_raw, xpl_raw)

st.caption(f"Pair key={pair.key} | dx={dx:.2f}, dy={dy:.2f} px")

# PREVIEW for fast slider
ppl_prev = resize_max_width(ppl_aligned, preview_w)
xpl_prev = resize_max_width(xpl, preview_w)

wipe_pct = st.slider("Wipe (left=PPL, right=XPL)", 0, 100, 50)
#st.image(wipe(ppl_prev, xpl_prev, wipe_pct), use_container_width=True)

st.image(
    wipe(ppl_prev, xpl_prev, wipe_pct),
    width=min(512, st.session_state.get("container_width", 512))
)

if show_full_side_by_side:
    combined = np.hstack([ppl_aligned, xpl])
    st.image(combined, caption="Full resolution: Left PPL (aligned) | Right XPL", use_container_width=True)
