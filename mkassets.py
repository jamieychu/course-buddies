"""Container asset prep: slug -> project file (strip _, space and - on BOTH sides;
the uploads for 3b are hyphenated, older ones are not), strict trim of
right/bottom pure-white padding, then clear the seam by flood-filling
near-white (>=228) from right/bottom-edge components seeded at >=240."""
import re, os, glob, sys, numpy as np
from PIL import Image
from scipy import ndimage
SRC = "/mnt/project"; OUT = "assets"
files = {}
for f in os.listdir(SRC):
    b, e = os.path.splitext(f)
    if e.lower() in (".png", ".jpg", ".jpeg"):
        files.setdefault(b.lower().replace("_", "").replace(" ", "").replace("-", ""), f)
# Strict trim only, no near-white flood: these assets are near-white
# FILL-TO-EDGE, so the flood eats the artwork. The Japan flag's field is
# (238,238,238) and reaches every edge -- the feed flag was rendering as a
# bare red disc in the container until this was spotted (3b, Sept 11).
# Container-only: Jamie's originals are transparent PNGs and never see this.
STRICT_ONLY = {"course-flag-upper-nav-icon", "course-score-flag-feed-icon"}
# Opt-in (3b, Sept 11): clear the recompression ring left against the padding.
# ~50 other assets carry the same ring; left alone because their screens were
# verified with it (a global switch would move pixels in verified renders).
SEAM_RING = {"active-continue-button", "secondary-active-cta-button"}
PATH_NODE = re.compile(r"course-icon|course-chest|rapid-review|score-6-icon")
def prep(slug, f):
    a = np.asarray(Image.open(os.path.join(SRC, f)).convert("RGB")).astype(int)
    y1, x1 = a.shape[:2]
    pure = (a >= 250).all(2)
    while x1 > 1 and pure[:y1, x1-1].all(): x1 -= 1
    while y1 > 1 and pure[y1-1, :x1].all(): y1 -= 1
    a = a[:y1, :x1]
    alpha = np.full(a.shape[:2], 255, np.uint8)
    if slug not in STRICT_ONLY:
        near = (a >= 228).all(2); seed = (a >= 240).all(2)
        lbl, n = ndimage.label(near)
        edge = set(np.unique(lbl[:, -1])) | set(np.unique(lbl[-1, :]))
        if PATH_NODE.search(slug):   # flattened transparency: white on every side
            edge |= set(np.unique(lbl[:, 0])) | set(np.unique(lbl[0, :]))
        edge.discard(0)
        keep = [k for k in edge if seed[lbl == k].any()]
        alpha[np.isin(lbl, keep)] = 0
        # Recompression rings the padding: the column (row) against the right
        # (bottom) padding keeps whitish pixels, e.g. (219,255,255) or
        # (248,255,227), that miss the >=228 test and draw a light line. Peel
        # the last opaque column/row ONLY when every opaque pixel left in it is
        # whitish (min channel >= 200), so art is never touched. (3b, Sept 11)
        if slug in SEAM_RING:
            for axis in (0, 1):                      # 0: last column, 1: last row
                for _ in range(2):
                    op = alpha > 0
                    idx = np.where(op.any(axis))[0]
                    if not len(idx): break
                    k = idx[-1]
                    line = (slice(None), k) if axis == 0 else (k, slice(None))
                    sel = op[line]
                    if not (a[line][sel].min(1) >= 200).all(): break
                    alpha[line] = np.where(sel, 0, alpha[line])
    # Rapid-review character crops (stage 2, Sept 15): the crops are RGB,
    # flattened on the reference copy's page colour (23,30,38) — some on
    # white — so every character sat in a rectangle of the wrong dark. Key
    # the flat background to transparent by CONNECTED flood from the edges
    # (tolerance 5), never by global colour distance: the locked characters'
    # shadows (30,37,45) are only 7 levels off the background and must stay.
    # Then un-flatten the 1-px antialiased rim against that background.
    if re.search(r"rapid-review|tile-avatar|course-score-flag-feed-icon", slug):   # crops flattened on the page colour (the flag: for the milestone's blue)
        alpha = unflatten(a, alpha)
    Image.fromarray(np.dstack([a.astype(np.uint8), alpha])).save(f"{OUT}/{slug}.png")


def key_white_upload(slug, tol=225, rim=2, soft=60):
    """Uploads that reach the container white-flattened with a 2-3px
    near-white re-encode fringe (avatar-pair-clean, nudge-cta-icon; Jamie's
    originals are transparent). Flood the near-white connected to the
    edges, un-blend a 2px rim against white. Enclosed whites (eye
    highlights) stay."""
    src = os.path.join(SRC, f"{slug}.png")
    if not os.path.exists(src): return
    a = np.array(Image.open(src).convert("RGB")).astype(int); alpha = np.full(a.shape[:2], 255, np.uint8)
    white = (a >= tol).all(2); lbl, n = ndimage.label(white)
    edge = set(np.unique(lbl[0, :])) | set(np.unique(lbl[-1, :])) | set(np.unique(lbl[:, 0])) | set(np.unique(lbl[:, -1])); edge.discard(0)
    alpha[np.isin(lbl, list(edge))] = 0
    ring = ndimage.binary_dilation(alpha == 0, iterations=rim) & (alpha > 0)
    d = np.abs(a - np.array([255, 255, 255])).max(2).astype(float); al = np.clip(d / soft, 0, 1)
    for y, x in zip(*np.where(ring)):
        k = al[y, x]
        if k <= 0: alpha[y, x] = 0; continue
        a[y, x] = np.clip((a[y, x] - (1 - k) * 255) / k, 0, 255); alpha[y, x] = int(round(k * 255))
    Image.fromarray(np.dstack([a.astype(np.uint8), alpha])).save(f"{OUT}/{slug}.png")


def derive_countdown_clock():
    """A white clock glyph from countdownchip.png (the quest page's dark
    countdown chip, "(clock) 23H"): the glyph crop (x 0-42, y 0-42 at 3x),
    the chip's dark fill keyed by connected flood, the opaque pixels painted
    white — the same derivation as close-x-white. The app always pairs the
    countdown with the clock (Jamie, Sept 15)."""
    src = os.path.join(SRC, "countdownchip.png")
    if not os.path.exists(src): return
    a = np.array(Image.open(src).convert("RGB").crop((0, 0, 42, 42))).astype(int)
    # the glyph is a light grey on a dark fill and its dial interior is
    # enclosed by the ring, so a connected flood can't reach it: alpha from
    # luminance instead (dark fill ~35 -> 0, glyph ~135 -> 1), then white
    lum = (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2])
    alpha = np.clip((lum - 45) / 70, 0, 1)
    alpha = (alpha * 255).round().astype(np.uint8)
    a[..., :] = 255
    Image.fromarray(np.dstack([a.astype(np.uint8), alpha])).save(f"{OUT}/countdown-clock.png")


def derive_two_friend_trim():
    """two-friend-figures: two-friend-icon-no-bg TRIMMED to its ink bounds
    (the upload carries ~120 px of transparent padding left and ~150 right,
    which made box-aligned placement land the ink 58 short; Jamie's own
    file may pad differently). With no padding the element's box IS the
    ink, so the Buddy Quest header aligns it by the box."""
    src = f"{OUT}/two-friend-icon-no-bg.png"
    if not os.path.exists(src): return
    im = Image.open(src).convert("RGBA"); al = np.array(im)[..., 3]; ys, xs = np.where(al > 0)
    im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).save(f"{OUT}/two-friend-figures.png")


def derive_contacts_icon():
    """The address-book icon from friendfindingoptions.png (the crop's row 1,
    device px x 36-194, y 38-197), keyed by connected flood against the
    tile fill so it shows NO fill on the rebuilt CSS tile (Jamie, Sept 15:
    the finding options are geometry, rebuilt; the illustration is cropped)."""
    src = os.path.join(SRC, "friendfindingoptions.png")
    if not os.path.exists(src): return
    a = np.array(Image.open(src).convert("RGB").crop((36, 38, 194, 197))).astype(int)
    alpha = np.full(a.shape[:2], 255, np.uint8)
    alpha = unflatten(a, alpha, tol=10)
    Image.fromarray(np.dstack([a.astype(np.uint8), alpha])).save(f"{OUT}/contacts-icon.png")


def derive_calendar_glyph():
    """A calendar glyph for the Co-Learning View's "days together" stat (Jamie,
    Sept 15): the "1" calendar from Streak_Menu_Reference's Streak Goal row
    (ref px x 61-119, y 634-700), its dark page keyed by connected flood."""
    src = os.path.join(SRC, "Streak_Menu_Reference.PNG")
    if not os.path.exists(src): return
    a = np.array(Image.open(src).convert("RGB").crop((61, 634, 119, 700))).astype(int)
    alpha = np.full(a.shape[:2], 255, np.uint8)
    alpha = unflatten(a, alpha, tol=12)
    Image.fromarray(np.dstack([a.astype(np.uint8), alpha])).save(f"{OUT}/calendar-glyph.png")


def derive_close_x_white():
    """close-x-white: the dismiss X for the quiz screens (white on pink).
    Jamie uploaded one on Sept 15; if it is in the folder it is keyed like
    any upload and this is skipped. Until then: the app's close-x glyph
    with its opaque pixels painted white, alpha kept — a colour change on
    a cropped glyph, not new artwork."""
    src = os.path.join(SRC, "close-x-white.png")
    if os.path.exists(src):
        key_white_upload("close-x-white")
        a = np.array(Image.open(f"{OUT}/close-x-white.png").convert("RGBA"))
        if (a[..., 3] > 40).sum() > 200: return
        # a white glyph flattened on WHITE by the upload keys to nothing (Sept 15: the 28 x 28 upload
        # arrived as 784 white pixels); fall through to the derivation from close-x
    base = f"{OUT}/close-x.png"
    if not os.path.exists(base): return
    a = np.array(Image.open(base).convert("RGBA"))
    a[..., :3] = 255
    Image.fromarray(a).save(f"{OUT}/close-x-white.png")


def derive_nudge_glyph():
    """The Nudge button's hand glyph, cropped from nudgecta.png (box measured
    Sept 15: x 128-198, y 36-108 at 3x). The bitmap button can't take a state
    and its label isn't Nunito, so the button is rebuilt in CSS around this
    glyph (build plan §12). The button fill is keyed to alpha with the
    same connected flood as the character crops."""
    src = os.path.join(SRC, "nudgecta.png")
    if not os.path.exists(src): return
    im = Image.open(src).convert("RGB").crop((128, 36, 198, 108))
    a = np.array(im).astype(int); alpha = np.full(a.shape[:2], 255, np.uint8)
    alpha = unflatten(a, alpha, tol=8)
    Image.fromarray(np.dstack([a.astype(np.uint8), alpha])).save(f"{OUT}/nudge-glyph.png")


def unflatten(a, alpha, tol=5, soft=40):
    """Clear the flat background connected to the crop's edges (or to already
    transparent padding); soften the rim. Background colour = the top-left
    pixel (dark page or white). Mutates a (rim colours) and returns alpha."""
    bg = a[0, 0].copy()
    if (bg >= 250).all():                       # white crop: the flood above did it
        return alpha
    near = (np.abs(a - bg).max(2) <= tol)
    # JPEG-ish noise breaks the background into islands (the surplus set 3
    # unlocked crops: 261 components); label a CLOSED copy so 1-px noise
    # gaps bridge, but only ever clear pixels that were near to begin with
    closed = ndimage.binary_closing(near, structure=np.ones((3, 3)), iterations=1) | near
    lbl, n = ndimage.label(closed)
    edge = set(np.unique(lbl[0, :])) | set(np.unique(lbl[-1, :])) | set(np.unique(lbl[:, 0])) | set(np.unique(lbl[:, -1]))
    edge |= set(np.unique(lbl[ndimage.binary_dilation(alpha == 0)]))   # adjacent to cleared padding counts as outside
    edge.discard(0)
    clear = np.isin(lbl, list(edge)) & near
    alpha[clear] = 0
    # rim: opaque pixels within 1 px of the cleared region get alpha from
    # their distance to the background and their colour un-blended
    rim = ndimage.binary_dilation(clear, iterations=1) & ~clear & (alpha > 0)
    d = np.abs(a - bg).max(2).astype(float)
    al = np.clip(d / soft, 0, 1)
    ys, xs = np.where(rim)
    for y, x in zip(ys, xs):
        k = al[y, x]
        if k <= 0: alpha[y, x] = 0; continue
        a[y, x] = np.clip((a[y, x] - (1 - k) * bg) / k, 0, 255)
        alpha[y, x] = int(round(k * 255))
    return alpha
if __name__ == "__main__":
    html = open("prototype.html").read()
    slugs = set(re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)+', html)) | {"scroll-icon-up", "scroll-icon-down"} | set(sys.argv[1:])
    made = []
    for s in sorted(slugs):
        k = s.replace("-", "")
        if k in files: prep(s, files[k]); made.append(s)
    print(len(made), "assets")

derive_nudge_glyph()
derive_close_x_white()
derive_countdown_clock()
derive_calendar_glyph()
derive_contacts_icon()
derive_two_friend_trim()
key_white_upload("avatar-pair-clean")
key_white_upload("course-score-duo-feed-icon")
key_white_upload("celebrate-score-duo")
key_white_upload("two-friend-icon-no-bg")
key_white_upload("monthly-badge-locked-icon")
key_white_upload("nudge-cta-icon")
