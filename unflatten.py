"""Key the flat background out of the rapid-review character crops.

Run on Jamie's originals if they are RGB (no alpha):
    python3 unflatten.py assets/rapid-review-*.png assets/rapidreview3stars.png
Each file is rewritten in place as RGBA. Skips files that already carry alpha.

Same algorithm as the container's mkassets.py (Sept 15): the background
colour is the top-left pixel (the reference copy's page #171E26, or white);
pixels within 5 levels of it that CONNECT to the crop's edges are cleared
(connectivity, not global colour distance — the locked characters' shadows
are only 7 levels off the background and must stay; a global key also
punches out the locked Duo's eyes); 1-px noise gaps are bridged; the 1-px
antialiased rim is un-blended against the background.
"""
import sys, numpy as np
from PIL import Image
from scipy import ndimage

def unflatten(a, alpha, tol=5, soft=40):
    bg = a[0, 0].copy()
    near = (np.abs(a - bg).max(2) <= tol)
    closed = ndimage.binary_closing(near, structure=np.ones((3, 3)), iterations=1) | near
    lbl, n = ndimage.label(closed)
    edge = set(np.unique(lbl[0, :])) | set(np.unique(lbl[-1, :])) | set(np.unique(lbl[:, 0])) | set(np.unique(lbl[:, -1]))
    edge |= set(np.unique(lbl[ndimage.binary_dilation(alpha == 0)]))
    edge.discard(0)
    clear = np.isin(lbl, list(edge)) & near
    alpha[clear] = 0
    rim = ndimage.binary_dilation(clear, iterations=1) & ~clear & (alpha > 0)
    d = np.abs(a - bg).max(2).astype(float); al = np.clip(d / soft, 0, 1)
    for y, x in zip(*np.where(rim)):
        k = al[y, x]
        if k <= 0: alpha[y, x] = 0; continue
        a[y, x] = np.clip((a[y, x] - (1 - k) * bg) / k, 0, 255); alpha[y, x] = int(round(k * 255))
    return alpha

for f in sys.argv[1:]:
    im = Image.open(f)
    if im.mode == "RGBA" and (np.asarray(im)[..., 3] == 0).any():
        print("skip (has alpha):", f); continue
    a = np.asarray(im.convert("RGB")).astype(int)
    alpha = np.full(a.shape[:2], 255, np.uint8)
    white = (a >= 250).all(2)                              # any white padding first
    lbl, n = ndimage.label(white)
    edge = set(np.unique(lbl[:, -1])) | set(np.unique(lbl[-1, :])) | set(np.unique(lbl[:, 0])) | set(np.unique(lbl[0, :])); edge.discard(0)
    alpha[np.isin(lbl, list(edge))] = 0
    alpha = unflatten(a, alpha)
    Image.fromarray(np.dstack([a.astype(np.uint8), alpha])).save(f)
    print("keyed:", f, "transparent px", int((alpha == 0).sum()))
