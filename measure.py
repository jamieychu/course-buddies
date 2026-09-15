"""Render vs reference, like-for-like. Template = reference patch over the
element's own art box (+2 CSS); NCC over the downsampled render; parabolic
sub-pixel peak. Returns render - ref in CSS px."""
import io, json, numpy as np, cv2
from PIL import Image
from playwright.sync_api import sync_playwright
S = 1.8404; URL = "file:///home/claude/build/prototype.html"
OVR = '()=>document.documentElement.style.setProperty("--surface-page","#171E26")'

def open_page(p):
    b = p.chromium.launch(); pg = b.new_page(viewport={"width":393,"height":852}, device_scale_factor=3)
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(700); return b, pg, errs

def grab(pg):
    im = Image.open(io.BytesIO(pg.screenshot())).convert("RGB").resize((723, 1568), Image.LANCZOS)
    return np.asarray(im).astype(np.float32)

def ref(name):
    return np.asarray(Image.open(f"/mnt/project/{name}").convert("RGB"))[:, :723].astype(np.float32)

def match(render, refimg, box, search=10):
    """box = (x0,y0,x1,y1) CSS screen coords. Template from ref, found in render."""
    x0, y0, x1, y1 = [int(round(v * S)) for v in box]
    y0 = max(y0, 0); y1 = min(y1, 1568)
    t = refimg[y0:y1, x0:x1]
    X0, Y0 = max(x0 - search, 0), max(y0 - search, 0)
    reg = render[Y0:y1 + search, X0:x1 + search]
    r = cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, (px, py) = cv2.minMaxLoc(r)
    def sub(a, i):
        if 0 < i < len(a) - 1:
            d = a[i-1] - 2*a[i] + a[i+1]
            return i + (0.5 * (a[i-1] - a[i+1]) / d if d else 0)
        return i
    sx = sub(r[py, :], px); sy = sub(r[:, px], py)
    return (X0 + sx - x0) / S, (Y0 + sy - y0) / S, mx
