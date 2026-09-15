"""Rendered-geometry tripwires — one landmark per screen.

Catches a WRONG ASSET (wrong resolution, wrong crop, missing file), not
fidelity. Tolerances are wide on purpose. Measured from the rendered
pixels, never from the DOM: an asset at the wrong resolution still gets the
DOM position it was given, but its artwork lands somewhere else.

Usage:  python3 tripwires.py [path/to/prototype.html]
Exit code 1 if any tripwire fails. Does not touch the page; nothing blocks.

Adding a screen: one entry in CHECKS — a preset, optional setup JS, and a
measure function returning named values in CSS px.
"""
import os, sys, numpy as np
from scipy import ndimage
from playwright.sync_api import sync_playwright
from PIL import Image
import io

URL = "file://" + (sys.argv[1] if len(sys.argv) > 1 else "/home/claude/build/prototype.html")
DPR = 3


def shot(pg):
    return np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB")).astype(int)


def largest(mask, y0, y1, x0=0, x1=393):
    """Largest connected component inside a CSS-px window -> bbox in CSS px."""
    Y0, Y1, X0, X1 = (int(v * DPR) for v in (y0, y1, x0, x1))
    m = ndimage.binary_fill_holes(mask[Y0:Y1, X0:X1])
    lbl, n = ndimage.label(m)
    if n == 0:
        return None
    k = 1 + np.argmax(ndimage.sum(m, lbl, range(1, n + 1)))
    ys, xs = np.where(lbl == k)
    return ((X0 + xs.min()) / DPR, (Y0 + ys.min()) / DPR,
            (X0 + xs.max() + 1) / DPR, (Y0 + ys.max() + 1) / DPR)


def banner(colour_test):
    def f(a, pg):
        b = largest(colour_test(a), 100, 205)
        return {"width": b[2] - b[0], "top": b[1]} if b else {}
    return f


blue = lambda a: (a[..., 2] > 170) & (a[..., 0] < 130)                          # banner face + edge
green = lambda a: (a[..., 1] > 140) & (a[..., 0] < 160) & (a[..., 2] < 110)


def trophy(a, pg):
    """A1 locked trophy: last node. Scrolled to the bottom so the nav can't clip it."""
    st = pg.evaluate("document.getElementById('scroller').scrollTop")
    bg = a[int(700 * DPR), int(8 * DPR)]                                         # empty gutter
    m = np.abs(a - bg).sum(2) > 60
    b = largest(m, 741.1 - st - 45, 741.1 - st + 45, 150, 245)
    return {"cx": (b[0] + b[2]) / 2, "cy": (b[1] + b[3]) / 2 + st} if b else {}


yellow = lambda a: (a[..., 0] > 200) & (a[..., 1] > 150) & (a[..., 2] < 120)


def feed_plate(a, pg):
    """Dani's flame, baked in the feed plate: catches a plate at the wrong scale/offset."""
    b = largest(yellow(a), 420, 535, 270, 370)
    return {"cx": (b[0] + b[2]) / 2, "cy": (b[1] + b[3]) / 2} if b else {}


def feed_pair(a, pg):
    """The avatar pair that replaced the Duo (Sept 11). Probe: Otis's purple
    avatar disc, which only the pair contains -- the Duo's green would also
    match the partner's light-green disc, so a green probe cannot tell the two
    assets apart (it silently latched onto it when the Duo was swapped out).
    Width included because the pair is CSS-sized by width, so a wrong width is
    the likely mistake and a centre alone would nearly survive it."""
    p = ((a[..., 2] > 150) & (a[..., 0] > 100) & (a[..., 0] < 190) & (a[..., 1] < 130))
    b = largest(p, 181, 243, 264, 366)
    out = {"corner": corner_delta(a)}
    if b: out.update(cx=(b[0] + b[2]) / 2, cy=(b[1] + b[3]) / 2, w=b[2] - b[0])
    return out


def feed_flag(a, pg):
    """The flag's white field: its width catches a top row left at the
    reference's own pixel size (the Sept 11 rescale is 44.22, was 31.01), which
    no probe on the pair would notice. It also catches the container's asset
    prep eating the field: this flag is near-white and fill-to-edge, so the
    seam flood cleared it and the flag rendered as a bare red disc until
    course-score-flag-feed-icon went into mkassets' STRICT_ONLY."""
    b = largest((a > 200).all(2), 143, 190, 262, 332)
    out = {}
    if b: out.update(w=b[2] - b[0], h=b[3] - b[1], cx=(b[0] + b[2]) / 2)
    g = largest((a > 200).all(2), 143, 192, 326, 362)          # the "6" glyph
    if g: out.update(num_h=g[3] - g[1], num_cx=(g[0] + g[2]) / 2)
    return out


def corner_delta(a):
    """Max channel difference between the pair's top-left corner and the card
    fill. 0 while the disc clip holds; the crop's baked background shows there
    the moment it doesn't, which no landmark probe would notice (the discs do
    not move). Container reads ~3 unclipped, a native render more."""
    box = lambda x0, y0, x1, y1: np.median(
        a[int(y0 * DPR):int(y1 * DPR), int(x0 * DPR):int(x1 * DPR)].reshape(-1, 3), 0)
    return float(np.abs(box(270, 188, 276, 193) - box(280, 250, 350, 258)).max())


def menu_panel(a, pg):
    """Panel bottom = where the 50% scrim starts, read down an empty gutter column."""
    col = a[int(560 * DPR):int(660 * DPR), int(5 * DPR)].mean(1)   # window moved with the panel (was 500-600)
    top, low = col[:20].mean(), col[-20:].mean()
    mid = (top + low) / 2
    i = int(np.argmax(col < mid))
    y = i - 1 + (col[i - 1] - mid) / (col[i - 1] - col[i])                       # interpolated crossing
    return {"bottom": 560 + (y + 0.5) / DPR}


# Feed targets read off the A3 render after it was verified: plate vs
# Friend_Quest_Post_Feed_Reference (11 landmarks, <= 0.10 CSS), Duo vs
# Feed_Course_Score_Reference (-0.19 / -0.26 after the stated transfer).
FP = (312.0, 477.2)            # feed plate: Dani's flame
FD = (336.83, 211.00, 41.00)   # avatar pair, clipped and 92 wide: purple disc
                               # centre and width (was the Duo's green blob)
pale = lambda a: (a[..., 2] > 230) & (a[..., 0] > 130) & (a[..., 0] < 185) & (a[..., 1] > 190)   # S2 extra-unit banner


def node_at(cx, cy):
    """Largest blob in a window round a path node (content coords). Rings sit
    clear of the node, so the node body is the largest component."""
    def f(a, pg):
        st = pg.evaluate("document.getElementById('scroller').scrollTop")
        bg = a[int(700 * DPR), int(8 * DPR)]
        m = np.abs(a - bg).sum(2) > 60
        b = largest(m, cy - st - 38, cy - st + 38, cx - 40, cx + 40)
        return {"cx": (b[0] + b[2]) / 2, "cy": (b[1] + b[3]) / 2 + st} if b else {}
    return f


def b2_star(a, pg):
    """B2 at rest: her in-progress star's disc (largest green blob; the ring
    segments are darker and separate), SCREEN coords. Catches a wrong star
    crop or a broken anchor rule (the rule puts this at a fixed screen y)."""
    b = largest(green(a), 406.85 - 45, 406.85 + 45, 126.62 - 50, 126.62 + 50)
    return {"cx": (b[0] + b[2]) / 2, "cy": (b[1] + b[3]) / 2} if b else {}


def rowblue(a):
    """Device px of selected-row blue in the Japanese row band. ~0 unselected."""
    b = a[int(590 * DPR):int(646 * DPR), int(8 * DPR):int(386 * DPR)]
    return float(((b[..., 2] > 130) & (b[..., 0] < 120) & (b[..., 1] > 95) & (b[..., 1] < 175)).sum())


def cs_open(a, pg):
    """Course selection, opening (disabled) state. JA flag's red disc = the row
    crop (wrong resolution or recrop moves/resizes it); the disabled pill = the
    plate window moved by SHIFT (a wrong shift or plate scale moves it)."""
    red = (a[..., 0] > 200) & (a[..., 1] < 110) & (a[..., 2] < 110)
    f = largest(red, 595, 640, 25, 85)
    bg = a[int(700 * DPR), int(5 * DPR)]
    p = largest(np.abs(a - bg).sum(2) > 40, 688, 750, 8, 386)
    out = {"row_blue": rowblue(a)}     # the two sources are geometrically identical,
    if f: out.update(disc_cx=(f[0] + f[2]) / 2, disc_cy=(f[1] + f[3]) / 2)   # so only colour
    if p: out.update(pill_top=p[1], pill_bot=p[3])                           # tells them apart
    return out


def cs_selected(a, pg):
    """Course selection after tapping Japanese. CONTINUE face (bright green,
    label filled) = the continue crop; the green ring (darker green, interior
    filled) = the secondary crop."""
    face = (a[..., 1] > 170) & (a[..., 0] > 110) & (a[..., 0] < 190) & (a[..., 2] < 110)
    c = largest(face, 685, 752, 8, 386)
    ring = (a[..., 1] > 105) & (a[..., 1] < 160) & (a[..., 0] > 70) & (a[..., 0] < 125) & (a[..., 2] < 80)
    r = largest(ring, 746, 812, 8, 386)
    out = {"row_blue": rowblue(a)}
    if c: out.update(cont_top=c[1], cont_right=c[2])   # right, not left: a wrong-resolution crop keeps its top-left
    if r: out.update(ring_top=r[1], ring_bot=r[3])
    return out


# Course selection targets, read off the verified 3b render (Sept 11): row crop
# +0.03/0.00 vs Flow_3's Japanese row; disabled pill +0.15/-0.06 vs Flow_2 moved
# -59.89; CONTINUE +0.06..+0.20/-0.01 vs Flow_3; ring ends +0.07/+0.08 vs the
# green reference. Intended: pill 698.10-742.11, ring 753.11-802.11.
CSO = (54.0, 616.83, 698.0, 742.0)        # disc cx, cy; pill top, bottom
CSS_ = (694.33, 377.0, 753.33, 802.0)     # CONTINUE top, right; ring top, bottom (intended right 377.06)
# Selected-row blue in the row band: ~0 unselected (23 px of the flag's edge),
# ~22.7k selected. Loose tolerances -- this separates the two sources, which are
# geometrically identical, so it only has to tell "some blue" from "none".

def marker(a, pg):
    """Partner marker (stage 2, Sept 13). Probes, all colour: the pin's white
    body (only white object in the window) and Otis's purple disc inside it.
    The disc's centre must sit on the pin body's centre, so a pin without its
    avatar fails. h = the white ink's HEIGHT against the intended 40 (body 32
    + tail 8; the >=250 threshold trims the antialiased crown row to 39.3):
    a pin painted over by the sticky banner measures shorter, while every
    position check reads its topmost VISIBLE pixel and passes. That is how
    a tangent crown got through 44 checks. rho = the pin's nearest ink to
    the start ring, in ring-ellipse units: below 1.0 the pin overlaps the
    ring, which the tip-only rule allowed at low shoulder angles."""
    w = (a >= 250).all(-1)
    b = largest(w, 180, 250, 215, 290)   # x from 215: the start node's white star ends at 213
    p = ((a[..., 2] > 150) & (a[..., 0] > 100) & (a[..., 0] < 190) & (a[..., 1] < 130))
    d = largest(p, 180, 250, 215, 290)
    out = {}
    if b:
        out.update(top=b[1], tip=b[3], h=b[3] - b[1], w=b[2] - b[0], bcx=(b[0] + b[2]) / 2)
        Y0, Y1, X0, X1 = (int(v * DPR) for v in (180, 250, 215, 290))
        ys, xs = np.where(w[Y0:Y1, X0:X1])
        X, Y = (X0 + xs) / DPR, (Y0 + ys) / DPR
        fx, fy = 196.63, 247.20 - 3.8                    # first node's face centre
        out["rho"] = float(np.sqrt(((X - fx) / 43.44) ** 2 + ((Y - fy) / 39.72) ** 2).min())
    if d: out.update(dcx=(d[0] + d[2]) / 2, dw=d[2] - d[0])
    if b and d: out["dxc"] = (d[0] + d[2]) / 2 - (b[0] + b[2]) / 2
    return out


def marker_taps(a, pg):
    """Real clicks, split by shape: the round body opens the Co-Learning View
    (the real overlay now, not a hook), the tail forwards to the node, and
    the node's own top-right shoulder under the tail still reaches the node."""
    out = {}
    pg.evaluate("()=>{closePopup()}")
    bx, by = pg.evaluate("()=>{const r=document.querySelector('#path .marker .hit-body').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}")
    pg.mouse.click(bx, by); pg.wait_for_timeout(150)
    out["body_clv"] = int(pg.evaluate("()=>SCENARIO.overlay==='colearning' && document.getElementById('colearn').classList.contains('open')")); out["body_popup"] = int(pg.evaluate("()=>SCENARIO.overlay==='firstNodePopup'"))
    pg.evaluate("()=>setState({overlay:'none'})"); pg.wait_for_timeout(120)
    tx, ty = pg.evaluate("()=>{const r=document.querySelector('#path .marker .tail-hit').getBoundingClientRect(); return [r.left+r.width/2, r.bottom-3]}")
    pg.mouse.click(tx, ty); pg.wait_for_timeout(120)
    out["tail_popup"] = int(pg.evaluate("()=>SCENARIO.overlay==='firstNodePopup'")); out["tail_clv"] = int(not pg.evaluate("()=>document.getElementById('colearn').classList.contains('open')"))
    pg.evaluate("()=>closePopup()")
    nx, ny = pg.evaluate("()=>{const r=document.querySelector('#path img[data-slot=\"node0\"]').getBoundingClientRect(); return [r.left+r.width*0.85, r.top+r.height*0.25]}")
    pg.mouse.click(nx, ny); pg.wait_for_timeout(120)
    out["shoulder_popup"] = int(pg.evaluate("()=>SCENARIO.overlay==='firstNodePopup'"))
    pg.evaluate("()=>closePopup()")
    return out


def marker_off(a, pg):
    w = (a >= 250).all(-1)
    return {"white": int(w[180 * DPR:240 * DPR, 215 * DPR:280 * DPR].sum())}


def rho_min(w, win, ellipses):
    """Smallest ring-ellipse radius over white ink in win (y0,y1,x0,x1, CSS)
    against each ellipse (cx, cy, rx, ry): < 1.0 means ink inside it."""
    Y0, Y1, X0, X1 = (int(v * DPR) for v in win)
    ys, xs = np.where(w[Y0:Y1, X0:X1]); X, Y = (X0 + xs) / DPR, (Y0 + ys) / DPR
    out = {}
    for name, (cx, cy, rx, ry) in ellipses.items():
        out[name] = float(np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2).min()) if len(X) else float("nan")
    return out


def marker_mid(a, pg):
    """Marker on node 1 of S1U1 (mid-path). Its preferred side is right, toward
    node 0's start ring; the any-node rule sends it left. rho_n0 measures the
    pin's ink against node 0's RING, rho_n2 against node 2's face, rho_own
    against node 1's own face: all must stay > 1 with the 4px gap
    (ring: 1 + 4/43 = 1.09; face: 1 + 4/34 = 1.12 on the x axis, less on y)."""
    w = (a >= 250).all(-1)
    b = largest(w, 240, 310, 95, 165)
    out = {}
    if b: out.update(top=b[1], h=b[3] - b[1], bcx=(b[0] + b[2]) / 2)
    r = rho_min(w, (240, 310, 95, 165), {
        "n0": (196.63, 247.20 - 3.8, 43.44, 39.72),
        "n2": (126.57, 408.61 - 3.8, 34, 27),
        "own": (149.46, 323.10 - 3.8, 34, 27)})
    # one-sided: viol = how far (in ellipse units) the ink gets inside the
    # 4px gap; 0 when clear. 4/43.44 = 0.092 on the ring, 4/34 = 0.118 on a face.
    out["viol_n0"] = max(0.0, 1.09 - r["n0"]); out["viol_n2"] = max(0.0, 1.10 - r["n2"]); out["viol_own"] = max(0.0, 1.10 - r["own"])
    out["rho_n0"] = r["n0"]
    return out


def bubble(slot, tail):
    """Direction bubble in a slot (y0,y1,x0,x1): white pin blob, Otis's disc
    centred on the body, and the avatar UPRIGHT — hair (dark) above the
    shirt (light): the top third of the disc is darker than the bottom
    third. A container flip that took the avatar with it inverts that."""
    def m(a, pg):
        w = (a >= 250).all(-1)
        b = largest(w, *slot)
        p = ((a[..., 2] > 150) & (a[..., 0] > 100) & (a[..., 0] < 190) & (a[..., 1] < 130))
        d = largest(p, *slot)
        out = {"btn": int(pg.evaluate("()=>document.getElementById('scrollbtn').style.display==='block'"))}
        if b: out.update(top=b[1], bot=b[3], h=b[3] - b[1], bcx=(b[0] + b[2]) / 2)
        if d: out.update(dcx=(d[0] + d[2]) / 2, dw=d[2] - d[0])
        if b:
            cx = (b[0] + b[2]) / 2; cy = b[1] + 16 if tail == "down" else b[3] - 16     # body centre from the pin box
            X0, X1, Y0, Y1 = (int(v * DPR) for v in (cx - 13, cx + 13, cy - 13, cy + 13))
            disc = a[Y0:Y1, X0:X1].mean(-1); n = disc.shape[0] // 3
            out["upright"] = int(disc[:n].mean() < disc[-n:].mean())
        if b and d: out["dxc"] = (d[0] + d[2]) / 2 - (b[0] + b[2]) / 2
        return out
    return m


def bubble_off(a, pg):
    w = (a >= 250).all(-1)
    return {"white_top": int(w[188 * DPR:245 * DPR, 321 * DPR:377 * DPR].sum()),
            "white_bot": int(w[685 * DPR:741 * DPR, 321 * DPR:377 * DPR].sum())}


def forks(a, pg):
    """Out-of-scope forks (Jamie, Sept 14): each active-but-unwired CTA beside
    a working one opens the PROTOTYPE card with its own string; the card hangs
    off the tapped CTA (above, or below when above would hit the scenario bar
    or the working CTA) with its caret on the CTA's centre and never covers
    the CTA; a tap anywhere dismisses it and still does its own thing."""
    S = {"friendA": "Not in this demo — tap + Course", "addB": "Not in this demo — tap LEARN WITH A FRIEND",
         "continue": "Not in this demo — tap WITH A FRIEND", "legendary": "Not in this demo — tap SIMULATE REVIEW"}
    def click(sel, dx=0.5, dy=0.5):
        # a missing element is a failed step, not a crashed suite
        r = pg.evaluate(f"()=>{{const e=document.querySelector('{sel}'); if(!e) return null; const r=e.getBoundingClientRect(); return [r.left+r.width*{dx}, r.top+r.height*{dy}]}}")
        if not r: return False
        pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(120); return True
    oos = lambda: pg.evaluate("()=>{const o=document.getElementById('oos'); return [o.classList.contains('open'), o.querySelector('.msg').textContent, o.dataset.side]}")
    def placed(sel, side, tsel):
        """card side as expected, caret x on the tapped CTA's centre (within 1),
        card rect clear of the tapped CTA, and — the rule that ranks first —
        the TARGET CTA's box (tsel) entirely clear of the card"""
        return pg.evaluate(f"""()=>{{const e=document.querySelector('{sel}'), t=document.querySelector('{tsel}'); const o=document.getElementById('oos'); if(!e||!t||!o.classList.contains('open')) return 0;
          const q=e.getBoundingClientRect(), c=o.getBoundingClientRect(), T=t.getBoundingClientRect(); const d=o.querySelector('path').getAttribute('d');
          const m=d.match(/L([0-9.]+),([0-9.]+) L/); const tipX=c.left+parseFloat(m[1])-1; const cx=(q.left+q.right)/2;
          const clear = c.bottom - 8 <= q.top + 0.5 || c.top + 8 >= q.bottom - 0.5;
          const sized = T.width > 10 && T.height > 10;
          const targetClear = sized && (c.left >= T.right || c.right <= T.left || c.top >= T.bottom || c.bottom <= T.top);
          return (o.dataset.side==='{side}' && Math.abs(tipX-cx) <= 1 && clear && targetClear) ? 1 : 0}}""")
    out = {}
    pg.evaluate("()=>applyPreset('A0')"); pg.wait_for_timeout(200)
    pg.evaluate("()=>{SCENARIO.overlay='courseMenu'; renderMenu()}"); pg.wait_for_timeout(200)
    click("#menu-art .cta-hit"); o = oos(); out["friendA"] = int(o[0] and o[1] == S["friendA"]); out["friendA_placed"] = placed("#menu-art .cta-hit", "below", '#menu-art img[data-slot="add-course-menu-icon"]')   # below: above would cover + Course
    pg.mouse.click(100, 300); pg.wait_for_timeout(120); out["friendA_dismiss"] = int(not oos()[0] and pg.evaluate("()=>SCENARIO.overlay") == "courseMenu")
    click('#menu-art img[data-slot="add-course-menu-icon"]'); pg.wait_for_timeout(200); out["addA_works"] = int(pg.evaluate("()=>SCENARIO.overlay") == "courseSelection")
    click("#coursesel .hit.cont"); out["continue_inactive_silent"] = int(not oos()[0])
    click("#coursesel .hit"); pg.wait_for_timeout(150); click("#coursesel .hit.cont"); o = oos(); out["continue"] = int(o[0] and o[1] == S["continue"]); out["continue_placed"] = placed("#coursesel .hit.cont", "above", "#coursesel .grp .sec-plate")
    pg.mouse.click(200, 400); pg.wait_for_timeout(120); out["continue_dismiss"] = int(not oos()[0] and pg.evaluate("()=>document.getElementById('coursesel').classList.contains('open')"))
    pg.evaluate("()=>applyPreset('B1')"); pg.wait_for_timeout(200); pg.evaluate("()=>{SCENARIO.overlay='courseMenu'; renderMenu()}"); pg.wait_for_timeout(200)
    click('#menu-art img[data-slot="add-course-menu-icon"]'); o = oos(); out["addB"] = int(o[0] and o[1] == S["addB"]); out["addB_placed"] = placed('#menu-art img[data-slot="add-course-menu-icon"]', "above", "#menu-art .cta-hit")
    pg.mouse.click(100, 300); pg.wait_for_timeout(120)
    pg.evaluate("()=>applyPreset('B7')"); pg.wait_for_timeout(200); pg.evaluate("()=>tapNode('node0')"); pg.wait_for_timeout(200)
    click("#popup .hit.oos"); o = oos(); out["legendary"] = int(o[0] and o[1] == S["legendary"]); out["legendary_placed"] = placed("#popup .hit.oos", "below", '#popup > *:not([style*="display: none"]) .hit:not(.oos)')
    # the working REVIEW must be uncovered while the card is up
    out["review_uncovered"] = pg.evaluate("""()=>{const o=document.getElementById('oos').getBoundingClientRect(); const r=document.querySelector('#popup > *:not([style*="display: none"]) .hit:not(.oos)').getBoundingClientRect();
        return (o.top >= r.bottom || o.bottom <= r.top) ? 1 : 0}""")
    # dismiss with a tap on the review popup's own title: the card closes and
    # the popup stays (a tap on the page would close the popup too, as it
    # always has — the card does not swallow taps)
    click('#popup > *:not([style*="display: none"]) img', 0.3, 0.15); out["legendary_dismiss"] = int(not oos()[0] and pg.evaluate("()=>SCENARIO.overlay") == "firstReviewPopup")
    click('#popup > *:not([style*="display: none"]) .hit:not(.oos)'); pg.wait_for_timeout(200); out["review_works"] = int(pg.evaluate("()=>SCENARIO.reviewCleared") == 1)
    out["no_scrim"] = int(pg.evaluate("()=>getComputedStyle(document.getElementById('oos')).backgroundColor") in ("rgba(0, 0, 0, 0)", "transparent"))
    out["fill_raised"] = int(pg.evaluate("()=>{const f=document.querySelector('#oos path').getAttribute('fill'); return f==='var(--surface-raised)'}"))
    out["frame_white"] = int(pg.evaluate("()=>getComputedStyle(document.documentElement).getPropertyValue('--proto-frame').trim().toUpperCase()") == "#FFFFFF")
    out["shadow"] = int(pg.evaluate("()=>getComputedStyle(document.querySelector('#oos .frame')).filter.startsWith('drop-shadow(')"))
    # leave the page clean for the next check: no overlays, no popup, no OOS
    pg.evaluate("()=>{document.getElementById('oos').classList.remove('open'); SCENARIO.overlay='none'; SCENARIO.courseSelected=null; renderCourseSel(); renderMenu(); renderPopup()}")
    return out


def score6_label(a, pg):
    """Score 6 simulate label white on the app's locked-grey plate."""
    c = pg.evaluate(r"""()=>{const p=[...document.querySelectorAll('#popup > *')].find(e=>e.style.display!=='none');
        const s=getComputedStyle(p.querySelector('.cta')).color;
        const m=s.replace('display-p3','').match(/[\d.]+/g).map(Number);
        return s.startsWith('color(') ? m.slice(0,3).map(v=>v*255) : m.slice(0,3)}""")
    return {"label_lum": (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])}


def ctile(a, pg):
    """Challenge tile in the rail (stage 2, Sept 15). Pixel probes: the band
    colour at the tile's centre line, the card face colour, and the gap
    from the screen-time tile's bottom to the tile's top (15). DOM: state,
    label, and the tap result — Locked opens the greyed popup with its
    caret on the tile's centre; Send fires the type-selection hook."""
    out = {}
    d = pg.evaluate("()=>{const t=document.getElementById('ctile'); const r=t.getBoundingClientRect(); return {on:t.classList.contains('on')?1:0, state:t.dataset.state, label:t.querySelector('span').textContent, left:r.left, top:r.top}}")
    out["on"] = d["on"]
    if not d["on"]: return out
    out.update(left=d["left"], top=d["top"], label_ok=int(d["label"] == {"locked": "Locked", "send": "Send", "answer": "Answer", "waiting": "Waiting"}[d["state"]]))
    band = a[int(332 * DPR), int(53 * DPR)]; card = a[int(294 * DPR), int(40 * DPR)]   # card probe left of the "?" knockout
    out["band_lum"] = float(0.2126 * band[0] + 0.7152 * band[1] + 0.0722 * band[2])
    out["band_pink"] = int(band[0] > 200 and band[2] > 150 and band[1] < 160)
    out["card_lum"] = float(0.2126 * card[0] + 0.7152 * card[1] + 0.0722 * card[2])
    # gap: the blue screen-time tile's last blue row above the tile's top
    col = a[int(240 * DPR):int(285 * DPR), int(53 * DPR)]
    blue = np.where((col[:, 2] > col[:, 0] + 40) & (col[:, 2] > 120))[0]
    if len(blue): out["gap"] = d["top"] - (240 + (blue.max() + 1) / DPR)
    # tap
    pg.mouse.click(d["left"] + 30, d["top"] + 30); pg.wait_for_timeout(150)
    pop = pg.evaluate("()=>{const p=document.getElementById('lockpop'); const r=p.getBoundingClientRect(); const c=p.querySelector('.caret').getBoundingClientRect(); return [p.classList.contains('open')?1:0, r.top, c.left+9]}")
    out["pop_open"] = pop[0]; out["typesel"] = int(pg.evaluate("()=>SCENARIO.overlay==='typeSelection'"))   # Send opens type selection
    if out["typesel"]: pg.evaluate("()=>closeTypeSelection()")
    if pop[0]:
        out["pop_top"] = pop[1]; out["caret_dx"] = pop[2] - (d["left"] + 59.7 / 2)
        out["pop_copy"] = int(pg.evaluate("()=>document.querySelector('#lockpop .msg').textContent") == "Both finish this unit to unlock quizzes!")   # Jamie's string, verbatim
        pg.mouse.click(200, 650); pg.wait_for_timeout(120); out["pop_closes"] = int(not pg.evaluate("()=>document.getElementById('lockpop').classList.contains('open')"))
    return out


def scenes(a, pg):
    """Paired character scenes (stage 2, Sept 15). DOM: the solo rapid-review
    image hidden and the pair (b behind, a in front, star row) shown when
    paired; the reverse when unpaired. Feet anchors as placed. A colour probe
    on set 1's Duo tells the locked and unlocked art apart (grey vs green)."""
    d = pg.evaluate("""()=>{const path=document.getElementById('path'); const pr=path.getBoundingClientRect(); const out={};
      for (const im of path.querySelectorAll('img[data-slot^="rr"]')) { const r=im.getBoundingClientRect();
        out[im.dataset.slot]={on: im.style.display!=='none' ? 1:0, art: im.dataset.art, l:r.left-pr.left, t:r.top-pr.top, b:r.bottom-pr.top, cx:(r.left+r.right)/2-pr.left}; }
      out.__scroll=document.getElementById('scroller').scrollTop; return out}""")
    out = {}
    solo = next((k for k in d if k in ("rr", "rr0")), None)
    if solo: out["solo_on"] = d[solo]["on"]
    for k in ("rra", "rrb", "rrs", "rr0a", "rr0b", "rr0s", "rr1a", "rr1b", "rr2a", "rr3a"):
        out[k + "_on"] = d[k]["on"] if k in d else 0      # a slot never created is off
    for k in ("rra", "rrb", "rrs", "rr0a", "rr1b", "rr2b", "rr3b"):
        if k in d and d[k]["on"]: out[k + "_cx"] = d[k]["cx"]; out[k + "_feet"] = d[k]["b"] if k != "rrs" else d[k]["t"]
    # set 3 is scaled 0.85 as a pair about the feet: both crops carry the same
    # scale, and the feet anchor does not move (b's bottom stays at 1850)
    # ...and set 4 carries the SAME factor: the same Bea appears in both sets on
    # one screen, and two scales of one character is what draws the eye
    for pre in ("rr2", "rr3"):
        sc = pg.evaluate(r"()=>[...document.querySelectorAll('#path img[data-slot^=\"%s\"]')].filter(e=>e.style.display!=='none').map(e=>{const m=e.style.transform.match(/scale\(([0-9.]+)\)/); return m?+m[1]:1})" % pre)
        if sc: out[pre + "_scale_ok"] = int(len(set(sc)) == 1 and abs(sc[0] - 0.85) < 0.001)
    for pre in ("rr", "rr0", "rr1", "rr2", "rr3"):
        A, B = d.get(pre + "a"), d.get(pre + "b")
        if A and B and A["on"] and B["on"]:
            ra = pg.evaluate(f"()=>{{const r=document.querySelector('#path img[data-slot=\"{pre}a\"]').getBoundingClientRect(); const q=document.querySelector('#path img[data-slot=\"{pre}b\"]').getBoundingClientRect(); return Math.max(q.left-r.right, r.left-q.right)}}")
            out[pre + "_gap_ok"] = int(ra >= 4)      # side by side, never overlapping
    if "rra" in d and d["rra"]["on"]:
        # the Duo's body colour just above its feet, at the screen position
        y = d["rra"]["b"] - 30 - d["__scroll"]; x = d["rra"]["cx"] + 6
        c = a[int(y * DPR), int(x * DPR)]
        out["duo_green"] = int(c[1] > 150 and c[1] > c[0] + 40 and c[1] > c[2] + 60)
        out["duo_grey"] = int(abs(int(c[0]) - int(c[1])) < 25 and abs(int(c[1]) - int(c[2])) < 25 and c[1] < 120)
    return out


def keyed(a, pg):
    """Rapid-review crops carry no background rectangle (Sept 15): in every
    rapid-review asset the (0,0) pixel is transparent and at most a rim's
    worth of background-coloured opaque pixels touch the transparent region.
    File-level, because the rectangle is invisible in a render at 2 levels
    off the page — which is how it hid through stage 1."""
    from scipy import ndimage
    from PIL import Image
    import glob
    worst = 0; corner = 0
    html = open(URL.replace("file://", "")).read() if URL.startswith("file://") else ""
    for f in glob.glob("assets/*rapid-review*.png"):
        slug = f.split("/")[-1][:-4]
        if html and slug not in html: continue      # derived files no state references (surplus set 3 unlocked)
        im = np.array(Image.open(f).convert("RGBA")).astype(int); al = im[..., 3]
        corner += int(al[0, 0] > 0)
        bgc = (np.abs(im[..., :3] - np.array([23, 30, 38])).max(2) <= 3) & (al > 0)
        worst = max(worst, int((bgc & ndimage.binary_dilation(al == 0, iterations=2)).sum()))
    return {"opaque_corners": corner, "rect_leftover": worst}


def typesel(a, pg):
    """Type selection (stage 2, Sept 15; §9) on the QUIZ BASE (Jamie): pink
    full-bleed, "Quiz your buddy" title, the card illustration, the §18
    tagline, white CTA. Four tiles in a 2x2 at the reference geometry,
    labels unique within the grid; tap selects (blue border) and lights the
    CTA; the CTA opens the question screen with challengeType set."""
    out = {}
    d = pg.evaluate("""()=>{const el=document.getElementById('typesel'); const tiles=[...el.querySelectorAll('.tile')].map(t=>{const r=t.getBoundingClientRect(); return {key:t.dataset.key, label:t.querySelector('.lab').textContent, l:r.left, t:r.top, w:r.width, h:r.height, sel:t.classList.contains('sel')}});
      const bg=getComputedStyle(el).backgroundColor; const ti=el.querySelector('.qtitle'); const tr=ti.getBoundingClientRect(); const il=el.querySelector('.qillo svg'); const ir=il?il.getBoundingClientRect():null;
      return {open:el.classList.contains('open'), off:el.querySelector('.qcta').classList.contains('off'), bg, title:ti.textContent, titleCy:tr.top+tr.height/2, illo:ir?[ir.left+ir.width/2, ir.top, ir.height]:null, tag:el.querySelector('.qtag').innerText, tiles}}""")
    out["open"] = int(d["open"]); out["n_tiles"] = len(d["tiles"]); out["tag_ok"] = int(d["tag"].replace("\n", " ") == "Answer five, then send them to Otis.")
    out["title_ok"] = int(d["title"] == "Quiz your buddy"); out["title_cy"] = d["titleCy"]
    c = re_rgb(d["bg"]); out["bg_tile_pink"] = int(abs(c[0] - 247) < 6 and abs(c[1] - 129) < 6 and abs(c[2] - 203) < 6)   # #F781CB, the tile's pink
    out["fan_of_five"] = int(pg.evaluate("()=>document.querySelectorAll('#typesel .fan svg').length") == 5)
    out["x_white"] = int("close-x-white" in pg.evaluate("()=>document.querySelector('#typesel .qx').getAttribute('srcset')"))
    c = centred(pg, "#typesel"); out["body_top"] = c["top"]; out["body_bottom"] = c["bottom"]; out["centred"] = int(c["gapDiff"] <= 1 and c["inside"] == 1)
    out["labels_unique"] = int(len(set(x["label"] for x in d["tiles"])) == len(d["tiles"]))
    if d["tiles"]:
        t0, t1, t2 = d["tiles"][0], d["tiles"][1], d["tiles"][2]
        out.update(t0_l=t0["l"], t0_t=t0["t"], t0_w=t0["w"], t0_h=t0["h"], gap_x=t1["l"] - (t0["l"] + t0["w"]), gap_y=t2["t"] - (t0["t"] + t0["h"]))
        # tiles stay dark on the pink base: probe a tile's fill just inside its border
        c = a[int((t0["t"] + t0["h"] - 8) * DPR), int((t0["l"] + 12) * DPR)]; out["tile_translucent"] = int(c[0] > 240 and 140 < c[1] < 175 and c[2] > 200)   # white 0.2 over the pink
        out["tile_no_dark"] = int(pg.evaluate("()=>getComputedStyle(document.querySelector('#typesel .tile')).backgroundColor").startswith("rgba(255, 255, 255"))
        out["no_outline"] = int(pg.evaluate("()=>getComputedStyle(document.querySelector('#typesel .tile:not(.sel)')).borderTopColor") in ("rgba(0, 0, 0, 0)", "transparent"))
        out["tag_two_lines"] = int(pg.evaluate("()=>document.querySelector('#typesel .qtag').innerText.split('\\n').length") == 2)
        im = pg.evaluate("()=>[...document.querySelectorAll('#typesel .tile img')].map(i=>Math.round(i.getBoundingClientRect().height))"); out["chars_equal_80"] = int(all(h == 80 for h in im) and len(im) == 4)
        lb = pg.evaluate("()=>[...document.querySelectorAll('#typesel .tile .lab')].map(l=>Math.round(l.getBoundingClientRect().top))"); out["label_areas_equal"] = int(lb[0] == lb[1] and lb[2] == lb[3])
    out["ready_before"] = int(not d["off"])
    pg.evaluate("()=>document.querySelector('#typesel .tile[data-key]').click()"); pg.wait_for_timeout(120)
    d2 = pg.evaluate("()=>{const el=document.getElementById('typesel'); const cta=el.querySelector('.qcta'); const cr=cta.getBoundingClientRect(); return {off:cta.classList.contains('off'), sel:[...el.querySelectorAll('.tile.sel')].length, ctaBg:getComputedStyle(cta).backgroundColor, cta:[cr.left,cr.top,cr.width,cr.height]}}")
    out["ready_after"] = int(not d2["off"]); out["one_selected"] = int(d2["sel"] == 1)
    cc = re_rgb(d2["ctaBg"]); out["cta_white"] = int(min(cc) > 240); out["cta_l"] = d2["cta"][0]; out["cta_t"] = d2["cta"][1]; out["cta_w"] = d2["cta"][2]
    r = pg.evaluate("()=>{const r=document.querySelector('#typesel .qcta').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}")
    pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(200)
    out["to_question"] = int(pg.evaluate("()=>SCENARIO.overlay==='question' && !!SCENARIO.challengeType"))
    pg.evaluate("()=>{SCENARIO.overlay='none'; SCENARIO.challengeType=null; render()}")
    return out


def question(kind):
    """Question screen (stage 2, Sept 15; §10) for one card: the capture used
    whole; five EMPTY segments in the track box, the captured fill hidden
    (the segment reads track grey, the gap reads page); energy shown as a
    number (25) over the capture's infinity, -5 on SIMULATE ANSWER ALL 5;
    the CTA relabelled in the capture's own CTA position (plate / keyboard
    plate / the speaking captures' deviation plate), label white."""
    def m(a, pg):
        out = {}
        d = pg.evaluate("""()=>{const el=document.getElementById('question'); const segs=[...el.querySelectorAll('.bar i')].map(s=>{const r=s.getBoundingClientRect(); return [r.left,r.top,r.width,r.height,s.classList.contains('on')?1:0]});
          const lab=el.querySelector('.ctalab'); const lr=lab.getBoundingClientRect(); const en=el.querySelector('.enum'); const er=en.getBoundingClientRect();
          return {open:el.classList.contains('open'), art:el.querySelector('.cap').dataset.art, cta:el.dataset.cta, segs, lab:lab.textContent, labTop:lr.top, labColor:getComputedStyle(lab).color, energy:en.textContent, energyL:er.left, energyCy:er.top+er.height/2}}""")
        out["open"] = int(d["open"]); out["cta_mode_ok"] = int(d["cta"] == kind); out["n_segs"] = len(d["segs"])
        if d["segs"]:
            s0, s4 = d["segs"][0], d["segs"][-1]
            out.update(bar_l=s0[0], bar_r=s4[0] + s4[2], bar_t=s0[1], bar_h=s0[3], on_count=sum(s[4] for s in d["segs"]))
            c1 = a[int((s0[1] + s0[3] / 2) * DPR), int((s0[0] + s0[2] / 2) * DPR)]
            out["seg1_grey"] = int(abs(int(c1[0]) - int(c1[1])) < 25 and c1[1] < 110 and c1[1] > 40)
            gx = s0[0] + s0[2] + 3; cg = a[int((s0[1] + s0[3] / 2) * DPR), int(gx * DPR)]
            out["gap_is_page"] = int(cg[1] < 60 and cg[0] < 60)     # neither green nor gold showing through
        out["lab_ok"] = int(d["lab"] == "SIMULATE ANSWER ALL 5"); out["lab_top"] = d["labTop"]
        out["plate_left"] = pg.evaluate("()=>document.querySelector('#question .ctaplate').getBoundingClientRect().left"); out["plate_h"] = pg.evaluate("()=>document.querySelector('#question .ctaplate').getBoundingClientRect().height")
        c = [int(v) for v in re_rgb(d["labColor"])]; out["lab_white"] = int(min(c) > 240)
        out["energy_25"] = int(d["energy"] == "25"); out["energy_l"] = d["energyL"]; out["energy_cy"] = d["energyCy"]
        # the infinity glyph must be gone: no white ink where it was (x 346-363, y 91-99) apart from the number's own blue
        reg = a[int(91 * DPR):int(100 * DPR), int(346 * DPR):int(364 * DPR)]; out["infinity_gone"] = int(int(((reg > 235).all(-1)).sum()) == 0)
        r = pg.evaluate("()=>{const r=document.querySelector('#question .ctahit').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}")
        pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(120); out["to_confirm"] = int(pg.evaluate("()=>SCENARIO.overlay==='confirmation'")); out["energy_after"] = pg.evaluate("()=>SCENARIO.energy")
        return out
    return m


def question_forks(a, pg):
    """The speaking capture's mic is a fork (out-of-scope popup pointing at
    the simulate plate); the gear opens the settings sheet; END SESSION
    returns to the path with the tile still Send; DONE is a fork."""
    out = {}
    def click(sel):
        r = pg.evaluate(f"()=>{{const e=document.querySelector('{sel}'); if(!e) return null; const r=e.getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}}")
        if not r: return False
        pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(150); return True
    oos = lambda: pg.evaluate("()=>{const o=document.getElementById('oos'); const r=o.getBoundingClientRect(); return [o.classList.contains('open'), o.querySelector('.msg').textContent, r.top, r.bottom]}")
    click("#question .michit"); o = oos(); out["mic_popup"] = int(o[0] and o[1] == "Not in this demo — tap SIMULATE ANSWER ALL 5")
    pl = pg.evaluate("()=>{const r=document.querySelector('#question .ctaplate').getBoundingClientRect(); return [r.top, r.bottom]}")
    out["mic_card_clear_of_plate"] = int(o[3] <= pl[0] or o[2] >= pl[1])
    mic = pg.evaluate("()=>document.querySelector('#question .michit').getBoundingClientRect().top")
    out["mic_card_gap"] = mic - o[3]      # card bottom (caret tip) to the mic's top edge: the placement gap, 4
    pg.mouse.click(300, 300); pg.wait_for_timeout(120)
    click("#question .gearhit"); out["gear_opens_sheet"] = int(pg.evaluate("()=>document.querySelector('#question .settings').classList.contains('open')"))
    click("#question .settings .donehit"); o = oos(); out["done_popup"] = int(o[0] and o[1] == "Not in this demo — tap END SESSION")
    en = pg.evaluate("()=>{const r=document.querySelector('#question .settings .endhit').getBoundingClientRect(); return [r.top, r.bottom]}")
    out["done_card_clear_of_end"] = int(o[3] <= en[0] or o[2] >= en[1])
    click("#question .settings .endhit"); pg.wait_for_timeout(200)
    out["end_to_path"] = int(pg.evaluate("()=>SCENARIO.overlay==='none' && !document.getElementById('question').classList.contains('open')"))
    out["tile_still_send"] = int(pg.evaluate("()=>document.getElementById('ctile').dataset.state==='send' && document.getElementById('ctile').classList.contains('on')"))
    return out


def sender_loop(a, pg):
    """The sender's loop, beats 12-17, as real taps (build plan §2b, §11,
    §12): SIMULATE ANSWER ALL 5 -> confirmation (hero 4, forward line,
    SEND; energy 20) -> SEND -> path with the tile Waiting -> tile -> status
    waiting (chip on, no report, no Nudge, control "answers") -> control ->
    status answered (five report rows, chip gone, Nudge on, control "sends
    one back") -> control -> path with the tile Answer."""
    out = {}
    def click(sel):
        r = pg.evaluate(f"()=>{{const e=document.querySelector('{sel}'); if(!e) return null; const r=e.getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}}")
        if not r: return False
        pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(200); return True
    click("#question .ctahit")
    d = pg.evaluate("()=>{const c=document.getElementById('confirm'); const n=c.querySelector('.hero .num'); const r=n.getBoundingClientRect(); const h=c.querySelector('.hero').getBoundingClientRect(); return {open:c.classList.contains('open'), num:n.textContent, cx:h.left+h.width/2, cy:h.top+h.height/2, dia:h.width, tag:c.querySelector('.qtag').textContent, cta:c.querySelector('.qcta .lab').textContent, energy:SCENARIO.energy}}")
    out["confirm_open"] = int(d["open"]); out["hero_4"] = int(d["num"] == "4"); out["hero_cx"] = d["cx"]; out["hero_dia"] = d["dia"]
    c = centred(pg, "#confirm"); out["confirm_centred"] = int(c["gapDiff"] <= 1 and c["inside"] == 1 and abs(c["bottom"] - 750.2) < 0.5)
    out["line_ok"] = int(d["tag"] == "Now see how Otis does."); out["cta_send"] = int(d["cta"] == "SEND"); out["energy_20"] = int(d["energy"] == 20)
    e = pg.evaluate("()=>{const c=document.getElementById('confirm'); const p=c.querySelector('.pairav img').getBoundingClientRect(); return {x:!!c.querySelector('.qx'), pairW:p.width, pairCx:p.left+p.width/2, s1:c.querySelector('.s1 .v .t').textContent, s2:c.querySelector('.s2') ? c.querySelector('.s2 .v .t').textContent : null, num:getComputedStyle(c.querySelector('.hero .num')).color}}")
    out["no_x"] = int(not e["x"]); out["pair_w"] = e["pairW"]; out["pair_cx"] = e["pairCx"]; out["row_type"] = int(e["s1"] == "Repeat after Bea"); out["no_combo_row"] = int(e["s2"] is None)
    f = pg.evaluate("()=>{const c=document.getElementById('confirm'); const p=c.querySelector('.pairav img').getBoundingClientRect(); const h=c.querySelector('.hero').getBoundingClientRect(); return {of:c.querySelector('.hero .of').textContent, k:c.querySelector('.s1 .k').textContent, gap:h.top-p.bottom, pairSrc:c.querySelector('.pairav img').getAttribute('srcset')}}")
    out["of_5"] = int(f["of"] == "of 5"); out["label_quiz_type"] = int(f["k"] == "Quiz type"); out["pair_ring_gap"] = f["gap"]; out["pair_clean"] = int("avatar-pair-clean" in f["pairSrc"])
    nc = re_rgb(e["num"]); out["num_pink"] = int(abs(nc[0] - 247) < 6 and abs(nc[1] - 129) < 6)
    # no red, no percentage on the confirmation
    txt = pg.evaluate("()=>document.getElementById('confirm').textContent"); out["no_percent"] = int("%" not in txt)
    click("#confirm .qcta")
    out["sent_to_path"] = int(pg.evaluate("()=>SCENARIO.overlay==='none' && SCENARIO.challengeState==='theirUnanswered'"))
    out["tile_waiting"] = int(pg.evaluate("()=>document.getElementById('ctile').querySelector('span').textContent==='Waiting'"))
    click("#ctile")
    d = pg.evaluate("()=>{const s=document.getElementById('status'); return {open:s.classList.contains('open'), state:s.dataset.state, rep:getComputedStyle(s.querySelector('.report')).display, rows:s.querySelectorAll('.report .rrow').length, nudge:getComputedStyle(s.querySelector('.nudge')).display, head:s.querySelector('.head').textContent, sim:s.querySelector('.sim .line').textContent, tag:s.querySelector('.sim .tag').textContent, type:s.querySelector('.s1 .v .t').textContent, countdown:!!s.querySelector('.chip'), simBg:getComputedStyle(s.querySelector('.sim')).backgroundColor, simBorder:getComputedStyle(s.querySelector('.sim')).borderTopColor, card:!!s.querySelector('.card')}}")
    out["status_open"] = int(d["open"]); out["waiting_state"] = int(d["state"] == "waiting"); out["no_countdown"] = int(not d["countdown"]); out["no_report"] = int(d["rep"] == "none" and d["rows"] == 0)
    out["no_nudge"] = int(d["nudge"] == "none"); out["head_waiting"] = int(d["head"] == "Waiting on Otis"); out["sim1"] = int(d["sim"] == "Simulate: Otis answers your quiz"); out["sim_tag"] = int(d["tag"] == "PROTOTYPE")
    out["type_row"] = int(d["type"] == "Repeat after Bea"); out["no_card"] = int(not d["card"])
    out["sim_on_colour"] = int(d["simBg"].startswith("rgba(255, 255, 255") and min(re_rgb(d["simBorder"])) > 240)
    w = pg.evaluate("()=>{const s=document.getElementById('status'); return {k:s.querySelector('.s1 .k').textContent, sub:s.querySelector('.sub').innerText.split('\\n').length, combo:getComputedStyle(s.querySelector('.split')).display}}")
    out["label_quiz_sent"] = int(w["k"] == "Quiz sent"); out["sub_two_lines"] = int(w["sub"] == 2); out["no_combo_waiting"] = int(w["combo"] == "none")
    c = centred(pg, "#status"); out["waiting_centred"] = int(c["gapDiff"] <= 1 and c["inside"] == 1 and abs(c["bottom"] - 828) < 0.5)   # no CTA: the band runs to 828
    g = pg.evaluate("()=>{const s=document.getElementById('status'); const h=s.querySelector('.head').getBoundingClientRect(), p=s.querySelector('.pairav img').getBoundingClientRect(); return {headFont:parseFloat(getComputedStyle(s.querySelector('.head')).fontSize), headAbovePair:h.bottom<=p.top?1:0, pairW:p.width, simH:s.querySelector('.sim').getBoundingClientRect().height}}")
    out["head_33"] = int(abs(g["headFont"] - 33) < 0.5); out["head_above_pair"] = g["headAbovePair"]; out["pair_240"] = int(abs(g["pairW"] - 240) < 1); out["sim_one_line"] = int(g["simH"] <= 44)
    x = pg.evaluate("()=>{const s=document.getElementById('status'); const sim=s.querySelector('.sim').getBoundingClientRect(); const kids=[...s.querySelector('.qbody').children].filter(c=>getComputedStyle(c).display!=='none'&&getComputedStyle(c).position!=='absolute'); const last=kids[kids.length-1].getBoundingClientRect(); return {simTop:sim.top, simClear:sim.top>=last.bottom?1:0, subFont:parseFloat(getComputedStyle(s.querySelector('.sub')).fontSize), x:s.querySelector('.qx').getAttribute('srcset')}}")
    out["sim_pinned"] = int(abs(x["simTop"] - 804) < 1 and x["simClear"] == 1); out["sub_21"] = int(abs(x["subFont"] - 21) < 0.5); out["x_white"] = int("close-x-white" in x["x"])
    xb2 = pg.evaluate("()=>{const r=document.querySelector('#status .qx').getBoundingClientRect(); return [r.left,r.top,r.width,r.height]}"); sh = shot_now(pg); x0, y0, w, h = [int(v * DPR) for v in xb2]; reg = sh[y0:y0 + h, x0:x0 + w]
    out["x_paints"] = int(int(((reg > 240).all(-1)).sum()) > 0.08 * w * h)   # PAINTED white ink, not a srcset
    xb = pg.evaluate("()=>{const r=document.querySelector('#status .qx').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2, r.width]}")
    out["x_cx"] = xb[0]; out["x_cy"] = xb[1]; out["x_w"] = xb[2]      # Match Madness's X: ink centre 28.07 / 86.7, 21.6 wide
    # the whole control is the tap target: tap its top-left corner, not the line
    r = pg.evaluate("()=>{const r=document.querySelector('#status .sim').getBoundingClientRect(); return [r.left+12, r.top+8]}"); pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(200)
    d = pg.evaluate("()=>{const s=document.getElementById('status'); const rows=[...s.querySelectorAll('.report .rrow')].map(r=>[r.classList.contains('ok')?1:0, r.querySelectorAll('.av img').length, getComputedStyle(r.querySelector('.an')).color, getComputedStyle(r).backgroundColor]); const n=s.querySelector('.nudge'); return {state:s.dataset.state, rep:getComputedStyle(s.querySelector('.report')).display, rows, nudge:getComputedStyle(n).display, nudgeBg:getComputedStyle(n).backgroundColor, nudgeCta:n.classList.contains('qcta'), nudgeTop:n.getBoundingClientRect().top, sim:s.querySelector('.sim .line').textContent, cs:SCENARIO.challengeState, head:s.querySelector('.head').textContent, simTop:s.querySelector('.sim').getBoundingClientRect().top, simBottom:s.querySelector('.sim').getBoundingClientRect().bottom, combo:s.querySelector('.split .half:last-child .v .t').textContent, comboShown:getComputedStyle(s.querySelector('.split')).display, sub:getComputedStyle(s.querySelector('.sub')).display, splitBelowRows:s.querySelector('.split').getBoundingClientRect().top>=s.querySelector('.report').getBoundingClientRect().bottom?1:0, halves:s.querySelectorAll('.split .half').length}}")
    out["sub_turn_state"] = int(d["sub"] != "none" and pg.evaluate("()=>document.querySelector('#status .sub').textContent") == "It's his turn to send one back."); out["stats_below_rows"] = d["splitBelowRows"]; out["split_two_halves"] = int(d["halves"] == 2)
    wd = pg.evaluate("()=>{const s=document.getElementById('status'); return [s.querySelector('.report').getBoundingClientRect().width, s.querySelector('.nudge').getBoundingClientRect().width, s.querySelector('.split').getBoundingClientRect().width, s.querySelector('.sim').getBoundingClientRect().width]}")
    out["one_margin"] = int(max(wd) - min(wd) < 0.5 and abs(wd[0] - 357.4) < 0.5)
    c = centred(pg, "#status"); out["answered_centred"] = int(c["gapDiff"] <= 1 and c["inside"] == 1)
    out["combo_1"] = int(d["combo"] == "1" and d["comboShown"] != "none")
    okc = [re_rgb(r[2]) for r in d["rows"] if r[0]]; missc = [re_rgb(r[2]) for r in d["rows"] if not r[0]]
    out["ok_text_pink"] = int(all(abs(c[0] - 247) < 6 and abs(c[1] - 129) < 6 for c in okc)); out["miss_text_white"] = int(all(min(c) > 240 for c in missc))
    out["no_green"] = int(not any("120, 201" in r[2] or "0.4706" in r[2] for r in d["rows"]))
    out["nudge_white_cta"] = int(d["nudgeCta"] and min(re_rgb(d["nudgeBg"])) > 240); out["nudge_above_control"] = int(d["nudgeTop"] < d["simTop"])
    out["nudge_in_cta_slot"] = int(abs(d["nudgeTop"] - 750.2) < 0.5); out["sim_pinned_answered"] = int(abs(d["simTop"] - 804) < 1)
    out["answered_state"] = int(d["state"] == "answered" and d["cs"] == "theirAnswered"); out["report_shown"] = int(d["rep"] == "block"); out["five_rows"] = int(len(d["rows"]) == 5)
    out["row_states_ok"] = int([r[:2] for r in d["rows"]] == [[1, 0], [0, 1], [0, 2], [1, 0], [1, 0]])      # fixture: both right 1,4,5; Otis missed 2; both missed 3
    out["nudge_on"] = int(d["nudge"] != "none"); out["sim2"] = int(d["sim"] == "Simulate: Otis sends one back"); out["head_answered"] = int(d["head"] == "Otis answered"); out["sim_fits"] = int(d["simBottom"] <= 852)
    click("#status .nudge"); out["nudge_fork"] = int(pg.evaluate("()=>document.getElementById('oos').classList.contains('open')")); pg.mouse.click(60, 60); pg.wait_for_timeout(100)
    click("#status .sim .line")
    out["your_turn"] = int(pg.evaluate("()=>SCENARIO.overlay==='none' && SCENARIO.challengeState==='yourTurn' && document.getElementById('ctile').querySelector('span').textContent==='Answer'"))
    return out


def feed_after_status(a, pg):
    """Switching from an open status screen to the feed closes every quiz
    overlay, and the feed's pair composite (stage 1) still renders from the
    ringed 840 x 476 avatar-pair-icon at its verified place."""
    pg.evaluate("()=>applyPreset('A16')"); pg.wait_for_timeout(200); pg.evaluate("()=>applyPreset('A3')"); pg.wait_for_timeout(300)
    d = pg.evaluate("()=>{const open=[...document.querySelectorAll('.quiz.open, #question.open')].length; const im=[...document.querySelectorAll('#feed img')].find(i=>i.src.includes('avatar-pair-icon')); const r=im?im.getBoundingClientRect():null; return {open, l:r?r.left:null, t:r?r.top:null, w:r?r.width:null, natural:im?im.naturalWidth:null}}")
    return {"overlays_open": d["open"], "pair_l": d["l"], "pair_t": d["t"], "pair_w": d["w"], "ringed_source": int(d["natural"] == 840)}


def _walk_click(pg, sel, wait=300):
    r = pg.evaluate(f"()=>{{const e=document.querySelector('{sel}'); if(!e) return null; const r=e.getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}}")
    if not r: return False
    pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(wait); return True


def walk_A(a, pg):
    """Preset A, 24 beats, every one a real tap from beat 1 (Jamie's 3c):
    no picker jumps mid-walkthrough."""
    out = {}; c = lambda sel, w=300: _walk_click(pg, sel, w)
    q = lambda js: pg.evaluate("()=>(" + js + ")")
    pg.evaluate("()=>applyPreset('A0')"); pg.wait_for_timeout(300)
    c("#stat-course img"); out["b2_course_menu"] = int(q("SCENARIO.overlay==='courseMenu'"))
    c('#menu-art img[data-slot="add-course-menu-icon"]'); out["b3_course_selection"] = int(q("SCENARIO.overlay==='courseSelection'"))
    c("#coursesel .hit"); c("#coursesel .hit.wf"); out["b4_friends"] = int(q("SCENARIO.overlay==='friends'"))
    c('#friends .row[data-k="otis"] .pill'); c("#friends .bcta"); out["b5_solo_path"] = int(q("CURRENT==='A1' && SCENARIO.overlay==='none' && SCENARIO.pairStatus!=='active'"))
    pg.wait_for_timeout(2300); out["b6_accepted"] = int(q("SCENARIO.overlay==='accepted'"))
    c("#accepted .bcta"); out["b7_layer"] = int(q("CURRENT==='A7' && SCENARIO.pairStatus==='active' && document.querySelector('.marker')!==null"))
    c(".marker .hit-body"); out["b8_clv"] = int(q("SCENARIO.overlay==='colearning'")); c("#colearn .qx")
    c('#path img[data-slot="node0"]'); out["b9_node_popup"] = int(q("SCENARIO.overlay==='firstNodePopup'"))
    c('#popup > *:not([style*="display: none"]) .hit:not(.oos)'); out["b10_path_send"] = int(q("CURRENT==='A10' && document.getElementById('ctile').querySelector('span').textContent==='Send'"))
    c("#ctile"); out["b11_typesel"] = int(q("SCENARIO.overlay==='typeSelection'"))
    c('#typesel .tile[data-key="bea"]'); c("#typesel .qcta"); out["b12_question"] = int(q("SCENARIO.overlay==='question'"))
    c("#question .ctahit"); out["b13_confirm"] = int(q("SCENARIO.overlay==='confirmation'"))
    c("#confirm .qcta"); out["b14_path_waiting"] = int(q("SCENARIO.overlay==='none' && document.getElementById('ctile').querySelector('span').textContent==='Waiting'"))
    c("#ctile"); out["b15_status_waiting"] = int(q("SCENARIO.overlay==='status' && document.getElementById('status').dataset.state==='waiting'"))
    c("#status .sim"); out["b16_answered"] = int(q("document.getElementById('status').dataset.state==='answered'"))
    c("#status .sim"); out["b17_path_answer"] = int(q("SCENARIO.overlay==='none' && document.getElementById('ctile').querySelector('span').textContent==='Answer'"))
    c("#ctile"); out["b18_state3"] = int(q("document.getElementById('status').dataset.state==='sentback'"))
    c("#status .answer"); out["b19_his_question"] = int(q("SCENARIO.overlay==='question'"))
    c("#question .ctahit"); out["b20_state4"] = int(q("document.getElementById('status').dataset.state==='youanswered'"))
    c("#status .notnow"); out["b21_path_send"] = int(q("SCENARIO.overlay==='none' && document.getElementById('ctile').querySelector('span').textContent==='Send'"))
    # the score-6 node: scroll it into view and tap it
    pg.evaluate("()=>{const s=document.getElementById('scroller'); s.scrollTop=s.scrollHeight}"); pg.wait_for_timeout(300)
    c('#path img[data-slot="score6"]'); out["b22_score6_popup"] = int(q("SCENARIO.overlay==='score6Popup'"))
    c('#popup > *:not([style*="display: none"]) .hit:not(.oos)'); out["b23_milestone"] = int(q("SCENARIO.overlay==='milestone' && SCENARIO.courses.japanese.score===6"))
    c("#milestone .qcta"); out["b23b_path"] = int(q("SCENARIO.overlay==='none' && SCENARIO.activeTab==='course'"))
    c('#nav img[data-slot="nav3"]'); out["b24_feed"] = int(q("SCENARIO.activeTab==='feed'"))
    return out


def walk_B(a, pg):
    """Preset B, 13 beats, every one a real tap from beat 1. Corrections to
    the inventory: the marker does NOT advance on SIMULATE REVIEW (Otis
    stays on the first node until REVIEW ALL); the friend row shows his
    course SCORE."""
    out = {}; c = lambda sel, w=300: _walk_click(pg, sel, w)
    q = lambda js: pg.evaluate("()=>(" + js + ")")
    pg.evaluate("()=>applyPreset('B2')"); pg.wait_for_timeout(300)   # beat 1: her solo path, further along, no rings, no pair
    out["b1_solo_no_rings"] = int(q("SCENARIO.pairStatus!=='active' && SCENARIO.reviewRange===null && [...document.querySelectorAll('#path .ring')].filter(r=>getComputedStyle(r).display!=='none').length===0"))
    c("#stat-course img"); out["b2_course_menu"] = int(q("SCENARIO.overlay==='courseMenu'"))
    c("#menu-art .cta-hit"); out["b3_friends"] = int(q("SCENARIO.overlay==='friends'"))
    out["b3_otis_score_shown"] = int(q("document.querySelector('#friends .row[data-k=\"otis\"] .sc .f')?.textContent==='10'"))   # equal to Jamie's 10
    c('#friends .row[data-k="otis"] .pill'); out["b4_sheet"] = int(q("SCENARIO.overlay==='startpoint' && !SCENARIO.invitePending"))
    c("#startpoint .bcta"); out["b4b_pending"] = int(q("SCENARIO.overlay==='friends' && SCENARIO.invitePending==='otis'"))
    c("#friends .bcta"); out["b5_her_path"] = int(q("CURRENT==='B2' && SCENARIO.overlay==='none' && [...document.querySelectorAll('#path .ring')].filter(r=>getComputedStyle(r).display!=='none').length===0"))
    pg.wait_for_timeout(2300); out["b6_accepted"] = int(q("SCENARIO.overlay==='accepted'"))
    c("#accepted .bcta"); out["b7_rings_and_marker"] = int(q("CURRENT==='B7' && SCENARIO.pairStatus==='active' && [...document.querySelectorAll('#path .ring')].filter(r=>getComputedStyle(r).display!=='none').length>0 && document.querySelector('.marker')!==null"))   # the layer paints: rings and the marker arrive with the pair
    c(".marker .hit-body"); out["b8_clv"] = int(q("SCENARIO.overlay==='colearning'")); c("#colearn .qx")
    c('#path img[data-slot="node0"]'); out["b9_first_ringed_popup"] = int(q("SCENARIO.overlay==='firstReviewPopup'"))
    c('#popup > *:not([style*="display: none"]) .hit:not(.oos)'); out["b10_review_marker_stays"] = int(q("SCENARIO.reviewCleared===1 && SCENARIO.partnerProgress===0"))
    last = pg.evaluate("()=>{const P=PATHS[`${SCENARIO.section}-${SCENARIO.unit}`]; const r=ringedNodes(P); return r[r.length-1]}")
    pg.evaluate("(i)=>{const im=document.querySelector('#path img[data-slot=\"node'+i+'\"]'); im.scrollIntoView({block:'center'})}", last); pg.wait_for_timeout(300)
    c(f'#path img[data-slot="node{last}"]'); out["b11_last_ringed_popup"] = int(q("SCENARIO.overlay==='lastReviewPopup'"))
    c('#popup > *:not([style*="display: none"]) .hit:not(.oos)'); out["b12_review_all"] = int(q("SCENARIO.partnerProgress===11 && SCENARIO.reviewCleared===15"))   # the range cleared; Otis moves to the extra unit
    cur = pg.evaluate("()=>Math.floor(SCENARIO.progress)")
    pg.evaluate("(i)=>{const im=document.querySelector('#path img[data-slot=\"node'+i+'\"]'); im.scrollIntoView({block:'center'})}", cur); pg.wait_for_timeout(300)
    c(f'#path img[data-slot="node{cur}"]'); out["b13_her_node_popup"] = int(q("SCENARIO.overlay!=='none'"))
    c('#popup > *:not([style*="display: none"]) .hit:not(.oos)'); out["b13_complete_lesson"] = int(q("SCENARIO.nodeLessons===2 && SCENARIO.reviewRange===null && SCENARIO.screenLock==='DONE'"))   # her star advances a segment as the range closes
    return out


def intro_popup(a, pg):
    """§17b: shown once on first open beneath preset A's beat 1; prototype
    chrome (white dashes on --surface-raised, the shadow), Duolingo-green
    title, pill and glyph icons, one line per line ("Scenario A" / "Both
    new to the language"), the quiet no-motion note; picking a row applies
    its beat 1 and the popup never returns. The picker: bar title in green,
    two scenario rows verbatim, a divider, Reset; only the selected row
    tinted with a green wash, border and check."""
    pg.reload(); pg.wait_for_timeout(900)
    d = pg.evaluate("""()=>{const i=document.getElementById('intro'); const c=i.querySelector('.card'); const cs=getComputedStyle(c); const rows=[...i.querySelectorAll('.row')].map(r=>[r.querySelector('.t').textContent, r.querySelector('.s').textContent, !!r.querySelector('.pill'), !!r.querySelector('.ico svg'), r.querySelector('.t').getBoundingClientRect().height<22?1:0, r.querySelector('.s').getBoundingClientRect().height<20?1:0]);
      const pill=i.querySelector('.pill'); const pr=pill.getBoundingClientRect(), tr=i.querySelector('.row[data-k="A"] .t').getBoundingClientRect();
      return {open:i.classList.contains('open'), under:CURRENT, dashed:cs.borderTopStyle, border:cs.borderTopColor, shadow:cs.boxShadow!=='none'?1:0, tag:i.querySelector('.tag').textContent, ttl:i.querySelector('.ttl').textContent, ttlColor:getComputedStyle(i.querySelector('.ttl')).color, sub:i.querySelector('.sub').textContent, rows, pillText:pill.textContent, pillColor:getComputedStyle(pill).color, pillClear:1, note:i.querySelector('.note').textContent, noteSize:parseFloat(getComputedStyle(i.querySelector('.note')).fontSize), pinkInChrome:(i.innerHTML+cs.color).toLowerCase().includes('f781cb')?1:0, squares:!!i.querySelector('.sq')}}""")
    out = {"open_on_load": int(d["open"] and d["under"] == "A0"), "frame_white_dashed": int(d["dashed"] == "dashed" and min(re_rgb(d["border"])) > 240 and d["shadow"] == 1), "tag": int(d["tag"] == "PROTOTYPE")}
    green = lambda c: abs(c[0] - 120) < 8 and abs(c[1] - 201) < 8 and abs(c[2] - 60) < 8
    out["title_green"] = int(d["ttl"] == "Course Buddies" and green(re_rgb(d["ttlColor"]))); out["sub_ok"] = int(d["sub"] == "Learn a course with a friend.")
    ln = pg.evaluate("""()=>{const i=document.getElementById('intro'); const lines=el=>{const cs=getComputedStyle(el); const lh=cs.lineHeight==='normal'?parseFloat(cs.fontSize)*1.364:parseFloat(cs.lineHeight); return Math.round(el.getBoundingClientRect().height/lh)};
      const tA=i.querySelector('.row[data-k="A"] .t'); const r=document.createRange(); r.selectNodeContents(tA); const pill=i.querySelector('.pill').getBoundingClientRect();
      return {w:i.querySelector('.card').getBoundingClientRect().width, sub:lines(i.querySelector('.sub')), t:[lines(tA), lines(i.querySelector('.row[data-k="B"] .t'))], s:[lines(i.querySelector('.row[data-k="A"] .s')), lines(i.querySelector('.row[data-k="B"] .s'))], note:lines(i.querySelector('.note')), clearance:pill.left-r.getBoundingClientRect().right}}""")
    out["card_313"] = int(abs(ln["w"] - 313) < 0.5); out["lines_held"] = int(ln["sub"] == 1 and ln["t"] == [1, 1] and ln["s"] == [1, 1] and ln["note"] == 1)
    # the tag on the corner: its box is above the title's INK (paint), and no title-white pixel lies inside it
    tg = pg.evaluate("()=>{const i=document.getElementById('intro'); const p=i.querySelector('.pill').getBoundingClientRect(); const rw=i.querySelector('.row[data-k=\"A\"]').getBoundingClientRect(); const rg=document.createRange(); rg.selectNodeContents(i.querySelector('.row[data-k=\"A\"] .t')); const t=rg.getBoundingClientRect(); return {p:[p.left,p.top,p.right,p.bottom], t:[t.left,t.top,t.right,t.bottom], corner:(Math.abs(p.right-rw.right)<=3 && p.top<rw.top+2)?1:0, radius:parseFloat(getComputedStyle(i.querySelector('.pill')).borderRadius)}}")
    sh = shot_now(pg); px = sh[int(tg["p"][1] * DPR):int(tg["p"][3] * DPR), int(tg["p"][0] * DPR):int(tg["p"][2] * DPR)]; tx = sh[int(tg["t"][1] * DPR):int(tg["t"][3] * DPR), int(tg["t"][0] * DPR):int(tg["t"][2] * DPR)]
    ink_top = (np.where((tx > 230).all(-1))[0].min() / DPR + tg["t"][1]) if ((tx > 230).all(-1)).any() else tg["t"][1]
    out["tag_on_corner"] = int(tg["corner"] == 1 and 4 <= tg["radius"] <= 6); out["tag_clears_title_ink"] = int(int(((px > 230).all(-1)).sum()) == 0 and tg["p"][3] <= ink_top)
    out["rows_verbatim_one_line"] = int(d["rows"] == [["Path A — Fresh Start", "Both new to the language", True, True, 1, 1], ["Path B — Meet Up", "Both already learning", False, True, 1, 1]] and not d["squares"])
    out["pill_start_here_green"] = int(d["pillText"] == "Start here" and green(re_rgb(d["pillColor"])) and d["pillClear"] == 1); out["note_quiet"] = int(d["note"] == "Simulate buttons skip ahead in time." and d["noteSize"] <= 11.5)
    out["no_pink_in_chrome"] = int(d["pinkInChrome"] == 0)
    r = pg.evaluate("()=>{const r=document.querySelector('#intro .row[data-k=\"B\"]').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}"); pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(300)
    out["pick_B_beat1"] = int(pg.evaluate("()=>CURRENT==='B2' && !document.getElementById('intro').classList.contains('open') && SCENARIO.reviewRange===null"))
    pk = pg.evaluate("()=>{document.getElementById('dropdown').classList.add('open'); const d=document.getElementById('dropdown'); const rows=[...d.querySelectorAll('.row')].map(r=>[r.dataset.k, r.querySelector('.t').textContent, r.querySelector('.s')?.textContent||null, r.classList.contains('on'), !!r.querySelector('.ico svg')]); const on=d.querySelector('.row.on'); const off=d.querySelector('.row:not(.on)[data-k]'); const bar=getComputedStyle(document.querySelector('#scenariobar span')).color; const r={bar, rows, hd:d.querySelector('.hd').textContent, sep:!!d.querySelector('.sep'), onBg:getComputedStyle(on).backgroundColor, onBorder:getComputedStyle(on).borderTopColor, tick:getComputedStyle(on.querySelector('.tick')).color, offBg:getComputedStyle(off).backgroundColor, offBorder:getComputedStyle(off).borderTopColor, barText:document.getElementById('v-scenario').textContent}; d.classList.remove('open'); return r}")
    out["bar_green"] = int(green(re_rgb(pk["bar"])) and pk["barText"] == "Path B — Meet Up"); out["selected_green"] = int(pk["onBg"].startswith("rgba(120, 201, 60") and pk["onBorder"].startswith("rgba(120, 201, 60") and green(re_rgb(pk["tick"])))
    out["others_neutral"] = int(pk["offBg"] in ("rgba(0, 0, 0, 0)", "transparent") and pk["offBorder"] in ("rgba(0, 0, 0, 0)", "transparent"))
    out["picker_collapsed"] = int(pk["rows"] == [["A0", "Path A — Fresh Start", "Both new to the language", False, True], ["B2", "Path B — Meet Up", "Both already learning", True, True], ["__reset", "Reset scenario", None, False, True]] and pk["sep"] and pk["hd"].upper() == "DEMO SCENARIOS")
    pg.evaluate("()=>applyPreset('A13')"); pg.wait_for_timeout(150); pg.evaluate("()=>{document.getElementById('dropdown').classList.add('open'); document.querySelector('#dropdown .row[data-k=\"__reset\"]').click()}"); pg.wait_for_timeout(200)
    out["reset_to_beat1"] = int(pg.evaluate("()=>CURRENT==='A0' && SCENARIO.overlay==='none'"))
    pg.evaluate("()=>applyPreset('B7')"); pg.wait_for_timeout(150); pg.evaluate("()=>{document.getElementById('dropdown').classList.add('open'); document.querySelector('#dropdown .row[data-k=\"__reset\"]').click()}"); pg.wait_for_timeout(200)
    out["reset_B_to_B2"] = int(pg.evaluate("()=>CURRENT==='B2' && SCENARIO.reviewRange===null && SCENARIO.pairStatus!=='active'"))
    return out


def feed_to_path(a, pg):
    """The feed tab opens from the paired path once the milestone has fired
    (score 6), and the course tab returns to THAT state — the layer, Otis at
    the rosette, the tile — not a solo path (Jamie). A tab switch changes
    activeTab only. The feed tab is not live from the solo A2."""
    def click(sel):
        r = pg.evaluate(f"()=>{{const r=document.querySelector('{sel}').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}}"); pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(250)
    pg.evaluate("()=>applyPreset('A21')"); pg.wait_for_timeout(200); click("#milestone .qcta")
    out = {"feed_tab_live": int(pg.evaluate("()=>document.querySelector('#nav img[data-slot=\"nav3\"]').style.cursor==='pointer'"))}
    click('#nav img[data-slot="nav3"]'); out["to_feed"] = int(pg.evaluate("()=>SCENARIO.activeTab==='feed'"))
    click('#nav img[data-slot="nav0"]')
    out["back_to_paired_path"] = int(pg.evaluate("()=>SCENARIO.activeTab==='course' && SCENARIO.pairStatus==='active' && SCENARIO.courses.japanese.score===6 && SCENARIO.partnerProgress==='score6' && document.querySelector('.marker')!==null && document.getElementById('ctile').classList.contains('on')"))
    pg.evaluate("()=>applyPreset('A3')"); pg.wait_for_timeout(200); click('#nav img[data-slot="nav0"]')
    out["A3_back_paired"] = int(pg.evaluate("()=>SCENARIO.activeTab==='course' && SCENARIO.pairStatus==='active' && document.querySelector('.marker')!==null"))
    pg.evaluate("()=>applyPreset('A2')"); pg.wait_for_timeout(200); out["solo_feed_not_live"] = int(pg.evaluate("()=>document.querySelector('#nav img[data-slot=\"nav3\"]').style.cursor!=='pointer'"))
    pg.evaluate("()=>{applyPreset('B7'); const c=JSON.parse(JSON.stringify(SCENARIO.courses)); c.japanese.score=6; setState({courses:c})}"); pg.wait_for_timeout(200)
    out["B_feed_never_live"] = int(pg.evaluate("()=>document.querySelector('#nav img[data-slot=\"nav3\"]').style.cursor!=='pointer'"))   # the feed carries Path A's post
    out["no_score_under_5"] = int(pg.evaluate("()=>FRIENDS.every(f=>Object.values(f.scores).concat(Object.values(f.scoresA||{})).every(v=>v>=5))"))   # a course score can't be 0; everyone starts at 5 — scoresA included
    return out


def dropdown(a, pg):
    """The picker dropdown: grouped under Preset A / Preset B, scrolls."""
    return pg.evaluate("""()=>{const d=document.getElementById('dropdown'); d.classList.add('open'); const hd=[...d.querySelectorAll('.hd')].map(h=>h.textContent);
      const r={hdA: hd[0]==='Preset A'?1:0, hdB: hd[1]==='Preset B'?1:0, scrolls: getComputedStyle(d).overflowY==='auto'?1:0, rows: d.querySelectorAll('.row[data-k]').length}; d.classList.remove('open'); return r}""")


def receiver_loop(a, pg):
    """States 3 and 4 (Jamie's four-state accounting), as real taps from
    A16: the control sends one back (Otis's type chosen at runtime, never
    Jamie's) -> path, tile Answer -> tile -> state 3 (no rows, no control,
    TIME LEFT 23H, ANSWER real) -> ANSWER -> question screen with his type
    -> SIMULATE ANSWER ALL 5 (energy 20 -> 15) -> state 4 (rows, split,
    SEND ONE BACK / NOT NOW, combo 2) -> NOT NOW -> path, tile Send."""
    out = {}
    def click(sel):
        r = pg.evaluate(f"()=>{{const e=document.querySelector('{sel}'); if(!e) return null; const r=e.getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}}")
        if not r: return False
        pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(200); return True
    pg.evaluate("()=>setState({challengeType:'falstaff'})"); pg.wait_for_timeout(100)     # Jamie sent Falstaff: Otis must not send it back
    click("#status .sim")
    d = pg.evaluate("()=>[SCENARIO.overlay, SCENARIO.challengeState, SCENARIO.challengeType, document.getElementById('ctile').querySelector('span').textContent]")
    out["sent_back"] = int(d[0] == "none" and d[1] == "yourTurn" and d[3] == "Answer"); out["type_differs"] = int(d[2] != "falstaff")
    click("#ctile")
    d = pg.evaluate("()=>{const s=document.getElementById('status'); return {st:s.dataset.state, rows:s.querySelectorAll('.report .rrow').length, rep:getComputedStyle(s.querySelector('.report')).display, sim:getComputedStyle(s.querySelector('.sim')).display, ans:getComputedStyle(s.querySelector('.answer')).display, big:s.querySelector('.split .half:last-child .big').textContent, k2:s.querySelector('.split .half:last-child .k').textContent, head:s.querySelector('.head').textContent}}")
    out["state3"] = int(d["st"] == "sentback" and d["head"] == "Otis sent one back"); out["s3_no_rows"] = int(d["rep"] == "none"); out["s3_no_control"] = int(d["sim"] == "none")
    out["s3_countdown"] = int(d["big"] == "23H" and d["k2"] == "Time left"); out["s3_answer_cta"] = int(d["ans"] == "block")
    ck = pg.evaluate("()=>{const c=document.querySelector('#status .split .half:last-child .v .clock'); const r=c.getBoundingClientRect(); return {on:getComputedStyle(c).display!=='none', nat:c.naturalWidth, box:[r.left,r.top,r.width,r.height]}}")
    shot = np.array(pg.screenshot(), dtype=np.uint8) if False else None
    b2 = shot_now(pg); x0, y0, w, h = [int(v * DPR) for v in ck["box"]]; reg = b2[y0:y0 + h, x0:x0 + w]
    out["s3_clock"] = int(ck["on"] and ck["nat"] > 0 and int(((reg > 240).all(-1)).sum()) > 0.2 * w * h)   # PAINTED: white ink in its box, not just an element with a srcset
    c = centred(pg, "#status"); out["s3_centred"] = int(c["gapDiff"] <= 1 and c["inside"] == 1 and abs(c["bottom"] - 750.2) < 0.5)
    click("#status .answer")
    d = pg.evaluate("()=>[SCENARIO.overlay, document.querySelector('#question .cap').dataset.art, SCENARIO.energy, SCENARIO.challengeType]")
    out["answer_to_question"] = int(d[0] == "question" and d[1].endswith(d[3]) or (d[0] == "question" and d[3] in d[1])); out["energy_before"] = d[2]
    click("#question .ctahit")
    d = pg.evaluate("()=>{const s=document.getElementById('status'); return {ov:SCENARIO.overlay, st:s.dataset.state, e:SCENARIO.energy, combo:SCENARIO.comboCount, cs:SCENARIO.challengeState, rows:s.querySelectorAll('.report .rrow').length, sb:getComputedStyle(s.querySelector('.sendback')).display, nn:getComputedStyle(s.querySelector('.notnow')).display, sim:getComputedStyle(s.querySelector('.sim')).display, big:s.querySelector('.split .half:last-child .big').textContent}}")
    out["state4"] = int(d["ov"] == "status" and d["st"] == "youanswered" and d["cs"] == "open"); out["energy_15"] = int(d["e"] == 15); out["combo_2"] = int(d["combo"] == 2 and d["big"] == "2")
    out["s4_rows"] = int(d["rows"] == 5); out["s4_footers"] = int(d["sb"] == "block" and d["nn"] == "block" and d["sim"] == "none")
    c = centred(pg, "#status"); out["s4_centred"] = int(c["gapDiff"] <= 1 and c["inside"] == 1)
    click("#status .notnow")
    out["notnow_to_path"] = int(pg.evaluate("()=>SCENARIO.overlay==='none' && document.getElementById('ctile').querySelector('span').textContent==='Send'"))
    pg.evaluate("()=>applyPreset('A20')"); pg.wait_for_timeout(200); click("#status .sendback")
    out["sendback_to_typesel"] = int(pg.evaluate("()=>SCENARIO.overlay==='typeSelection' && !document.getElementById('status').classList.contains('open') && document.elementFromPoint(196,400).closest('.quiz').id==='typesel'"))   # VISIBLY on top, not just the state
    pg.evaluate("()=>closeTypeSelection()")
    return out


def assets_load(a, pg):
    """Every assets/*.png the build references exists in the assets folder
    and loads in the page (naturalWidth > 0). Derived assets — the clock,
    the white X — are the ones a copied build can lack; a missing file
    shows as nothing at all."""
    import re, os
    html = open(URL.replace("file://", "")).read(); base = os.path.dirname(URL.replace("file://", ""))
    slugs = sorted(set(re.findall(r"assets/([A-Za-z0-9\-]+)\.png", html)))
    missing = [s for s in slugs if not os.path.exists(os.path.join(base, "assets", f"{s}.png"))]
    bad = pg.evaluate("(slugs)=>Promise.all(slugs.map(s=>new Promise(res=>{const i=new Image(); i.onload=()=>res(i.naturalWidth>0?null:s); i.onerror=()=>res(s); i.src='assets/'+s+'.png'}))).then(r=>r.filter(Boolean))", slugs)
    if missing or bad: print("      missing/unloadable assets:", missing, bad)
    return {"referenced": len(slugs), "missing": len(missing), "unloadable": len(bad)}


def shot_now(pg):
    """A fresh screenshot as an array, for checks that must read paint after
    their own taps."""
    from PIL import Image
    import io
    return np.array(Image.open(io.BytesIO(pg.screenshot())).convert("RGB")).astype(int)


def milestone(a, pg):
    """§15 on the blue base: centred between the top and the reference's
    CTA slot (683.4), Duo at its 1x size, flag + 6, the pair, the §18 line;
    CONTINUE closes onto the path; the share square and MORE ABOUT SCORE
    are forks. Fires from the score-6 simulate when paired."""
    d = pg.evaluate("""()=>{const m=document.getElementById('milestone'); const b=m.querySelector('.qbody').getBoundingClientRect(); const kids=[...m.querySelector('.qbody').children]; const r0=kids[0].getBoundingClientRect(), r1=kids[kids.length-1].getBoundingClientRect();
      const order=kids.map(k=>k.className.split(' ')[0]); return {order:order.join(','), open:m.classList.contains('open'), gapDiff:Math.abs((r0.top-b.top)-(b.bottom-r1.bottom)), duo:m.querySelector('.duo').getBoundingClientRect().width, n:m.querySelector('.score .n').textContent, flagW:m.querySelector('.score .flag').getBoundingClientRect().width, line:m.querySelector('.qtag').innerText.replace('\\n',' '), cta:m.querySelector('.qcta').getBoundingClientRect().top, bg:getComputedStyle(m).backgroundImage.startsWith('linear-gradient')?1:0, pair:m.querySelector('.pairav img').getBoundingClientRect().width}}""")
    out = {"open": int(d["open"]), "centred": int(d["gapDiff"] <= 1), "duo_168": int(abs(d["duo"] - 168) < 1), "six": int(d["n"] == "6"), "flag_w": d["flagW"], "order_ok": int(d["order"] == "pairav,score,duo,qtag"),
           "line_ok": int(d["line"] == "You and Otis both reached Japanese Score 6!"), "cta_top": d["cta"], "blue": d["bg"], "pair_w": d["pair"]}
    def click(sel):
        r = pg.evaluate(f"()=>{{const r=document.querySelector('{sel}').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}}"); pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(200)
    click("#milestone .share"); out["share_fork"] = int(pg.evaluate("()=>document.getElementById('oos').classList.contains('open')")); pg.mouse.click(100, 100); pg.wait_for_timeout(100)
    click("#milestone .qcta"); out["continue_to_path"] = int(pg.evaluate("()=>SCENARIO.overlay==='none' && SCENARIO.courses.japanese.score===6"))
    pg.evaluate("()=>applyPreset('A10')"); pg.evaluate("()=>{SCENARIO.overlay='score6Popup'; renderPopup(); POPUP.score6Popup.action()}"); pg.wait_for_timeout(300)
    out["fires_from_score6"] = int(pg.evaluate("()=>SCENARIO.overlay==='milestone' && document.getElementById('milestone').classList.contains('open')"))
    return out


def colearn(a, pg):
    """§8 as a surface with the reference's PINK header at the top: the X and
    "Japanese with Otis" over it, "Buddy Quest" with the clock countdown, the
    two characters right, the dark card with the quest line and a readable
    counter in the bar; then scores side by side, the turn ROW BUTTON that
    routes to the relevant challenge state, OVERVIEW with real icons (the
    combo hidden at zero), the badges strip, STOP LEARNING TOGETHER small
    and last. Month counter capped at 19 (prototype limit)."""
    d = pg.evaluate("""()=>{const c=document.getElementById('colearn'); const h=c.querySelector('.hdr'); const hr=h.getBoundingClientRect(); const bg=getComputedStyle(h).backgroundColor; const page=getComputedStyle(c).backgroundColor;
      const fig=h.querySelector('.pairfig').getBoundingClientRect(), hero=h.querySelector('.hero').getBoundingClientRect(), x=h.querySelector('.qx').getBoundingClientRect(); const bqr=h.querySelector('.bq').getBoundingClientRect();
      const xOwnRow=(x.bottom<=bqr.top && Math.abs(x.top-75)<0.5)?1:0; const noTitle=!h.querySelector('.title')?1:0;
      const shs=[...c.querySelectorAll('.sh')].map(e=>e.textContent.trim().toUpperCase()); const cards=[...c.querySelectorAll('.body .card')].map(e=>e.getBoundingClientRect().width);
      return {open:c.classList.contains('open'), bg, page, hdr:[hr.left, hr.top, hr.width, hr.height], xIn:(x.top>=hr.top && x.bottom<=hr.bottom)?1:0, xW:x.width, xSrc:h.querySelector('.qx').getAttribute('srcset'), xOwnRow, noTitle, bq:h.querySelector('.bq').textContent, bqFont:parseFloat(getComputedStyle(h.querySelector('.bq')).fontSize),
        cd:h.querySelector('.cd span').textContent, clock:h.querySelector('.cd img').naturalWidth, figRight:fig.right, figW:fig.width, figOnCard:(fig.bottom>hero.top && fig.top<hero.top)?1:0, heroW:hero.width, heroIn:(hero.bottom<=hr.bottom)?1:0,
        qn:h.querySelector('.qn').textContent, qnFont:parseFloat(getComputedStyle(h.querySelector('.qn')).fontSize), shs, cards, turn:c.querySelector('.turn .tt').textContent, chev:!!c.querySelector('.turn .chev'), turnCursor:getComputedStyle(c.querySelector('.turn')).cursor,
        days:c.querySelector('.ov b.days').textContent, daysIcon:c.querySelector('.ov .st-days img').naturalWidth, combo:getComputedStyle(c.querySelector('.ov .combo')).display, badges:c.querySelectorAll('.badges img').length, you:c.querySelector('.scores .you').textContent, otis:c.querySelector('.scores .otis').textContent,
        av:c.querySelector('.scores .p > img').getBoundingClientRect().height, sim:!!c.querySelector('.sim'), endFont:parseFloat(getComputedStyle(c.querySelector('.end')).fontSize), endLast:c.querySelector('.body').lastElementChild.className==='end'?1:0, ovWrap:[...c.querySelectorAll('.ov .st')].every(e=>e.getBoundingClientRect().height<30)?1:0}}""")
    pk = re_rgb(d["bg"]); pg_ = re_rgb(d["page"])
    out = {"open": int(d["open"]), "hdr_pink": int(abs(pk[0] - 247) < 6 and abs(pk[1] - 129) < 6), "hdr_full_bleed": int(d["hdr"][0] == 0 and d["hdr"][1] == 0 and abs(d["hdr"][2] - 393) < 0.5), "page_dark": int(max(pg_) < 60)}
    out["x_own_row_above"] = int(d["xIn"] and d["xOwnRow"] and "close-x-white" in d["xSrc"] and abs(d["xW"] - 23) < 0.5); out["no_header_text"] = d["noTitle"]
    xb2 = pg.evaluate("()=>{const r=document.querySelector('#colearn .hdr .qx').getBoundingClientRect(); return [r.left,r.top,r.width,r.height]}"); sh = shot_now(pg); x0, y0, w, h = [int(v * DPR) for v in xb2]; reg = sh[y0:y0 + h, x0:x0 + w]
    out["x_paints"] = int(int(((reg > 240).all(-1)).sum()) > 0.08 * w * h); out["buddy_quest"] = int(d["bq"] == "Buddy Quest" and abs(d["bqFont"] - 28) < 0.5)
    out["countdown"] = int(d["cd"] == "25 days" and d["clock"] > 0)
    ty = pg.evaluate("()=>{const c=document.getElementById('colearn'); const g=s=>c.querySelector(s).getBoundingClientRect(); const f=s=>parseFloat(getComputedStyle(c.querySelector(s)).fontSize); return {bq:f('.bq'), cd:f('.cd'), gapAbove:g('.bq').top-100, gapBetween:g('.cd').top-g('.bq').bottom, gapBelow:g('.hero').top-g('.cd').bottom, figTop:g('.pairfig').top, figW:g('.pairfig').width, cardTop:g('.hero').top, clock:g('.cd img').height, hdrH:g('.hdr').height, scoresTop:g('.scores').top, bqLeft:g('.bq').left, xBottom:g('.hdr .qx').bottom, bqTop:g('.bq').top}}")
    out["bq_28_cd_15"] = int(abs(ty["bq"] - 28) < 0.5 and ty["cd"] == 15 and abs(ty["bqLeft"] - 20) < 0.5 and ty["xBottom"] <= ty["bqTop"]); out["block_centred"] = int(abs(ty["gapAbove"] - ty["gapBelow"]) < 1 and ty["gapBetween"] >= 10)   # the stack below the X row, centred between it and the card
    # the band, the figures, the card and everything below are where they were (band 286, fig 104 at 142 wide, card 180, scores 302)
    out["band_figures_card_unmoved"] = int(abs(ty["hdrH"] - 286) < 0.5 and abs(ty["cardTop"] - 180) < 0.5 and abs(ty["figTop"] - 104) < 0.5 and abs(ty["figW"] - 142) < 0.5 and abs(ty["scoresTop"] - 302) < 0.5); out["clock_13"] = int(abs(ty["clock"] - 13) < 0.5)
    out["figures_right"] = int(d["figRight"] > 360 and d["figOnCard"] == 1); out["hero_in_header"] = int(d["heroIn"] == 1 and abs(d["heroW"] - 357.4) < 0.5)
    out["counter_readable"] = int(d["qn"] == "0 / 20" and d["qnFont"] >= 14); out["headers"] = int(d["shs"] == ["OVERVIEW", "MONTHLY BADGES"]); out["one_margin"] = int(all(abs(w - 357.4) < 0.5 for w in d["cards"]))
    lk = pg.evaluate("()=>{const t=document.querySelector('#colearn .turn'); return {locked:t.classList.contains('locked'), tt:t.querySelector('.tt').textContent, ts:t.querySelector('.ts').textContent, cursor:getComputedStyle(t).cursor, lock:getComputedStyle(t.querySelector('.lock')).display, chev:getComputedStyle(t.querySelector('.chev')).display}}")
    out["turn_locked_row"] = int(lk["locked"] and lk["tt"] == "Quizzes locked" and lk["ts"] == "Both finish Section 1, Unit 1 to unlock" and lk["cursor"] == "default" and lk["lock"] != "none" and lk["chev"] == "none")
    ol = pg.evaluate("()=>{const t=document.querySelector('#colearn .turn .ts'); const r=t.getBoundingClientRect(); const lk=document.querySelector('#colearn .turn .lock').getBoundingClientRect(); return {h:r.height, font:parseFloat(getComputedStyle(t).fontSize), clear:r.right<=lk.left-4?1:0, noClip:t.scrollWidth<=t.clientWidth+0.5?1:0}}")
    out["locked_copy_one_line"] = int(ol["h"] <= ol["font"] * 1.5 and ol["clear"] == 1 and ol["noClip"] == 1)   # one line, clear of the lock, nothing clipped
    r0 = pg.evaluate("()=>{const r=document.querySelector('#colearn .turn').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}"); pg.mouse.click(r0[0], r0[1]); pg.wait_for_timeout(200)
    out["turn_locked_inert"] = int(pg.evaluate("()=>SCENARIO.overlay==='colearning' && !document.getElementById('oos').classList.contains('open')"))
    out["days_1_icon"] = int(d["days"] == "1" and d["daysIcon"] > 0); out["ov_one_line"] = d["ovWrap"]
    sp = pg.evaluate("()=>{const c=document.getElementById('colearn'); const g=s=>c.querySelector(s).getBoundingClientRect(); const cal=c.querySelector('.ov .st-days img').getAttribute('srcset'); return {figRight:g('.pairfig').right, cardRight:g('.hero').right, within:parseFloat(getComputedStyle(c.querySelector('.ov')).rowGap), between:g('.sh').top-g('.turn').bottom, endGap:g('.end').top-g('.badges').bottom, cal}}")
    # the pair's INK right edge on the card's, read from paint (the trimmed asset's box is its ink; paint is still what's checked)
    b2 = shot_now(pg); band = b2[int(100 * DPR):int(180 * DPR), int(150 * DPR):int(391 * DPR)]; mk = (np.abs(band - np.array([247, 129, 203])).sum(-1) > 60); ys, xs = np.where(mk)
    ink_right = (xs.max() + 1) / DPR + 150; out["fig_ink_on_card_edge"] = int(abs(ink_right - sp["cardRight"]) < 1.5); out["fig_below_x_row"] = int((ys.min() / DPR + 100) >= 100)
    # the card's top crosses the figures between the shoulder and the bottom (the reference overlaps: faces clear, lower bodies behind the card)
    hero_top = pg.evaluate("()=>document.querySelector('#colearn .hero').getBoundingClientRect().top"); ink_top = ys.min() / DPR + 100; ink_bottom = 195.4   # the trimmed figures' bottom (under the card)
    # the card overlaps only the crop's bottom edge: faces clear (the chin sits ~62% down the ink), at most a fifth of the crop under the card
    out["card_over_bottom_edge_only"] = int(ink_top + 0.62 * (ink_bottom - ink_top) < hero_top and (ink_bottom - hero_top) <= 0.22 * (ink_bottom - ink_top) + 0.5)
    out["hdr_compact"] = int(pg.evaluate("()=>document.querySelector('#colearn .hdr').getBoundingClientRect().height") <= 290)
    out["sections_double"] = int(sp["between"] >= 2 * sp["within"] - 0.5 and sp["endGap"] >= 60); out["calendar_icon"] = int("calendar-glyph" in sp["cal"])
    ov = pg.evaluate("()=>[document.querySelector('#colearn .ov .st-days span').textContent, document.querySelector('#colearn .ov .st.combo > span:last-child').textContent, getComputedStyle(document.querySelector('#colearn .ov .combo')).display, document.querySelector('#colearn .ov .cmb').textContent]")
    out["day_singular"] = int(ov[0] == "day together"); out["combo_shown_at_zero"] = int(ov[2] != "none" and ov[3] == "0" and ov[1] == "quizzes in a row")
    out["four_badges"] = int(d["badges"] == 4); out["scores"] = int(d["you"] == "5" and d["otis"] == "5"); out["avatar_44"] = int(abs(d["av"] - 44) < 1); out["no_sim"] = int(not d["sim"]); out["end_small_last"] = int(d["endLast"] == 1 and d["endFont"] <= 12.5)
    def click_turn():
        r = pg.evaluate("()=>{const r=document.querySelector('#colearn .turn').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}"); pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(250)
    pg.evaluate("()=>applyPreset('A18')"); pg.evaluate("()=>setState({overlay:'colearning'})"); pg.wait_for_timeout(150); click_turn(); out["turn_to_state3"] = int(pg.evaluate("()=>SCENARIO.overlay==='status' && document.getElementById('status').dataset.state==='sentback'"))
    pg.evaluate("()=>applyPreset('A15')"); pg.evaluate("()=>setState({overlay:'colearning'})"); pg.wait_for_timeout(150); click_turn(); out["turn_to_waiting"] = int(pg.evaluate("()=>SCENARIO.overlay==='status' && document.getElementById('status').dataset.state==='waiting'"))
    pg.evaluate("()=>applyPreset('A10')"); pg.evaluate("()=>setState({overlay:'colearning'})"); pg.wait_for_timeout(150); click_turn(); out["turn_to_typesel"] = int(pg.evaluate("()=>SCENARIO.overlay==='typeSelection' && document.elementFromPoint(196,400).closest('.quiz').id==='typesel'")); pg.evaluate("()=>closeTypeSelection()")
    pg.evaluate("()=>applyPreset('A16')"); pg.wait_for_timeout(150)
    for _ in range(20): pg.evaluate("()=>{setState({challengeState:'theirUnanswered', overlay:'status'}); document.querySelector('#status .sim').click()}"); pg.wait_for_timeout(30)
    pg.evaluate("()=>setState({overlay:'colearning'})"); pg.wait_for_timeout(150)
    out["combo_tracks"] = int(pg.evaluate("()=>SCENARIO.comboCount") == 21); out["month_capped_19"] = int(pg.evaluate("()=>document.querySelector('#colearn .qn').textContent") == "19 / 20" and pg.evaluate("()=>SCENARIO.quizzesThisMonth") == 19)
    pg.evaluate("()=>applyPreset('B7')"); pg.wait_for_timeout(150); pg.evaluate("()=>setState({overlay:'colearning'})"); pg.wait_for_timeout(150)
    out["B_scores_10_10"] = int(pg.evaluate("()=>document.querySelector('#colearn .scores .you').textContent==='10' && document.querySelector('#colearn .scores .otis').textContent==='10'"))   # Path B: equal scores, a unit apart in position
    pg.evaluate("()=>applyPreset('A7')"); pg.wait_for_timeout(200)
    r = pg.evaluate("()=>{const r=document.querySelector('.marker .hit-body').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}"); pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(250)
    out["marker_opens"] = int(pg.evaluate("()=>SCENARIO.overlay==='colearning' && document.elementFromPoint(196,400).closest('#colearn')!==null"))
    r = pg.evaluate("()=>{const r=document.querySelector('#colearn .qx').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}"); pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(200)
    out["x_closes"] = int(pg.evaluate("()=>SCENARIO.overlay==='none'"))
    return out


def invite_flow(a, pg):
    """§5-§7 as real taps, both presets, on the app's dark surfaces. Friend
    selection: back arrow, the §18 title, the Japanese flag line, five rows
    (flags with numbers; the sort rule: Japanese learners by closeness to
    Jamie's score, non-Japanese after, the paired row greyed in place;
    Otis first in A at 0=0, third in B at 5 vs 10), the finding options
    BELOW the list, both pill states at one geometry, other INVITEs forks
    whose popup never covers Otis's row, CONTINUE disabled until Otis's
    INVITE flips his pill to PENDING and the others to disabled. CONTINUE:
    A -> the solo path, ~1s -> "Otis is in." (bubble, pair, the subheadline
    below the pair, CONTINUE) -> A7; B -> the bottom-packed start-point
    sheet (option 1 a fork) -> SEND INVITE -> B1 -> ~1s -> accepted -> B7."""
    out = {}
    def click(sel):
        r = pg.evaluate(f"()=>{{const e=document.querySelector('{sel}'); if(!e) return null; const r=e.getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}}")
        if not r: return False
        pg.mouse.click(r[0], r[1]); pg.wait_for_timeout(250); return True
    dark = lambda sel: max(re_rgb(pg.evaluate(f"()=>getComputedStyle(document.querySelector('{sel}')).backgroundColor"))) < 60
    rows_q = """()=>[...document.querySelectorAll('#friends .row')].map(r=>({k:r.dataset.k, ja:[...r.querySelectorAll('.sc .f')].filter(f=>f.querySelector('img').getAttribute('srcset')==='assets/course-flag-upper-nav-icon.png 3x').map(f=>+f.textContent)[0], nflags:r.querySelectorAll('.sc .f img').length, paired:r.classList.contains('paired'), pill:r.querySelector('.pill').className, pillW:r.querySelector('.pill').getBoundingClientRect().width, h:r.getBoundingClientRect().height}))"""
    # ---- preset A ----
    pg.evaluate("()=>applyPreset('A0')"); pg.wait_for_timeout(200); pg.evaluate("()=>{SCENARIO.overlay='courseSelection'; renderCourseSel()}"); pg.wait_for_timeout(200)
    click("#coursesel .hit"); click("#coursesel .hit.wf")
    d = pg.evaluate("()=>{const f=document.getElementById('friends'); return {open:f.classList.contains('open'), top:document.elementFromPoint(196,400).closest('#friends')?1:0, title:f.querySelector('.title').textContent, course:f.querySelector('.course span').textContent, flag:f.querySelector('.course img').naturalWidth, counter:/invites? remaining/i.test(f.textContent)?1:0, above:f.querySelector('.list').getBoundingClientRect().bottom<=f.querySelector('.finding').getBoundingClientRect().top?1:0, cta:f.querySelector('.bcta').classList.contains('off')?1:0}}")
    rows = pg.evaluate(rows_q)
    out["friends_open"] = int(d["open"] and d["top"]); out["friends_dark"] = int(dark("#friends")); out["title_ok"] = int(d["title"] == "Who do you want to learn with?"); out["course_flag"] = int(d["course"] == "Japanese" and d["flag"] > 0)
    out["no_counter"] = int(d["counter"] == 0); out["list_above_options"] = d["above"]; out["cta_off_before"] = d["cta"]; out["rows_5"] = int(len(rows) == 5); out["row_h"] = rows[0]["h"]
    fl0 = pg.evaluate("()=>[...document.querySelectorAll('#friends .row')].map(r=>[r.dataset.k, [...r.querySelectorAll('.sc .f')].map(f=>f.querySelector('img').getAttribute('srcset')+'|'+f.textContent)])")
    out["A_order"] = int([r[0] for r in fl0] == ["otis", "sabrina", "david", "alexandra", "angel"])
    out["A_otis_italian_5_no_japanese"] = int(rows[0]["k"] == "otis" and rows[0]["ja"] is None and fl0[0][1] == ["assets/italy-course-flag-icon.png 3x|5"])
    out["A_sabrina_spanish_8"] = int(fl0[1][1] == ["assets/spanish-course-flag-icon.png 3x|8"])
    live = [r for r in rows if not r["paired"]]; ja = [r["ja"] for r in live]; nonja = [i for i, v in enumerate(ja) if v is None]
    out["A_non_japanese_first"] = int(nonja == list(range(len(nonja))) and len(nonja) == 2)   # Jamie has none: the no-Japanese rows rank first, in fixture order
    withja = [v for v in ja if v is not None]; out["A_japanese_ascending"] = int(withja == sorted(withja) and withja == [16, 18])
    out["flags_not_words"] = int(all(r["nflags"] > 0 or r["paired"] for r in rows)); out["pills_equal"] = int(len(set(round(r["pillW"], 1) for r in rows)) == 1)
    pr = [r for r in rows if r["paired"]]; out["paired_greyed_last"] = int(len(pr) == 1 and "off" in pr[0]["pill"] and rows[-1]["paired"])
    ft = pg.evaluate("()=>{const f=document.getElementById('friends'); return {h:f.querySelector('.fh').textContent, tiles:[...f.querySelectorAll('.ftile')].map(t=>Math.round(t.getBoundingClientRect().height)), crop:!!f.querySelector('.finding img[srcset*=\"friend-finding-options\"]'), icon:f.querySelector('.ftile[data-k=\"contacts\"] img').naturalWidth}}")
    out["finding_css"] = int(ft["h"] == "Find your friends" and ft["tiles"] == [64, 64, 64] and not ft["crop"] and ft["icon"] > 0)
    # the contacts icon shows NO fill on its tile: read the tile's own fill just left of the icon's box, from paint
    b2 = shot_now(pg); ib = pg.evaluate("()=>{const r=document.querySelector('#friends .ftile[data-k=\"contacts\"] img').getBoundingClientRect(); const t=document.querySelector('#friends .ftile[data-k=\"contacts\"]').getBoundingClientRect(); return [r.left, r.top, r.width, r.height, t.left, t.top]}")
    x0, y0, w, h = [int(v * DPR) for v in ib[:4]]; inside = b2[y0:y0 + h, x0:x0 + w]; edge = b2[y0 + 2:y0 + h - 2, x0 - 6 * DPR:x0 - 2 * DPR]
    tilefill = b2[int((ib[5] + 8) * DPR), int((ib[4] + 40) * DPR)]
    out["contacts_no_fill"] = int(np.abs(edge - tilefill).max() < 12)
    # a fork beneath Otis: the popup sits below the tapped row and never covers his
    click('#friends .row[data-k="sabrina"] .pill')
    fk = pg.evaluate("()=>{const o=document.getElementById('oos'); const r=o.getBoundingClientRect(); const ot=document.querySelector('#friends .row[data-k=\"otis\"]').getBoundingClientRect(); return {open:o.classList.contains('open'), msg:o.querySelector('.msg').textContent, side:o.dataset.side, coversOtis:!(r.bottom<=ot.top||r.top>=ot.bottom)}}")
    out["fork_below_never_covers_otis"] = int(fk["open"] and fk["side"] == "below" and not fk["coversOtis"] and fk["msg"] == "Not in this demo — tap INVITE on Otis TM"); pg.mouse.click(200, 90); pg.wait_for_timeout(100)
    click('#friends .row[data-k="otis"] .pill')
    pd = pg.evaluate("()=>{const f=document.getElementById('friends'); return {pending:f.querySelector('.row[data-k=\"otis\"] .pill').textContent, others:[...f.querySelectorAll('.row:not([data-k=\"otis\"]) .pill')].every(p=>p.classList.contains('off')), cta:!f.querySelector('.bcta').classList.contains('off'), rows:f.querySelectorAll('.row').length}}")
    out["pending_state"] = int(pd["pending"] == "Pending" and pd["others"] and pd["cta"] and pd["rows"] == 5)
    click("#friends .bcta"); out["A_continue_to_path"] = int(pg.evaluate("()=>CURRENT==='A1' && SCENARIO.overlay==='none'"))
    pg.wait_for_timeout(2300); out["A_acceptance_arrives"] = int(pg.evaluate("()=>SCENARIO.overlay==='accepted' && document.getElementById('accepted').classList.contains('open')"))
    e = pg.evaluate("()=>{const a=document.getElementById('accepted'); const order=[...a.querySelector('.qbody').children].map(c=>c.className); const p=a.querySelector('.pairav img').getBoundingClientRect(); return {order:order.join(','), line:a.querySelector('.line').textContent, sub:a.querySelector('.sub').textContent, bubble:a.querySelector('.bubble span').textContent, subBelowPair:a.querySelector('.sub').getBoundingClientRect().top>=p.bottom?1:0, pairW:p.width, x:!!a.querySelector('.qx'), sim:!!a.querySelector('.sim'), cta:a.querySelector('.bcta .lab').textContent, ctaTop:a.querySelector('.bcta').getBoundingClientRect().top}}")
    out["accepted_A_copy"] = int(e["line"] == "Otis is in." and e["sub"] == "You're both starting from scratch!" and e["bubble"] == "Let's do this!"); out["sub_below_pair"] = e["subBelowPair"]; out["accepted_order"] = int(e["order"] == "line,bubble,pairav,sub")
    out["accepted_dark"] = int(dark("#accepted")); out["pair_264"] = int(abs(e["pairW"] - 264) < 2); out["no_x_no_sim"] = int(not e["x"] and not e["sim"]); out["cta_continue"] = int(e["cta"] == "CONTINUE" and abs(e["ctaTop"] - 693.7) < 0.5)
    click("#accepted .bcta"); out["continue_to_A7"] = int(pg.evaluate("()=>CURRENT==='A7' && SCENARIO.pairStatus==='active' && SCENARIO.overlay==='none' && !SCENARIO.invitePending"))
    # ---- preset B ----
    pg.evaluate("()=>applyPreset('B1')"); pg.wait_for_timeout(200); pg.evaluate("()=>{SCENARIO.overlay='courseMenu'; renderMenu()}"); pg.wait_for_timeout(200)
    click("#menu-art .cta-hit"); out["B_from_menu"] = int(pg.evaluate("()=>SCENARIO.overlay==='friends' && SCENARIO.friendsFrom==='courseMenu'"))
    rows = pg.evaluate(rows_q); ja = [r["ja"] for r in rows if r["ja"] is not None and not r["paired"]]
    out["B_otis_first_at_10"] = int(rows[0]["k"] == "otis" and rows[0]["ja"] == 10 and rows[-1]["paired"]); out["B_sorted"] = int(ja == sorted(ja, key=lambda v: abs(v - 10)) and ja[1:] == [16, 18])
    snap = lambda: pg.evaluate("()=>({overlay:SCENARIO.overlay, pending:SCENARIO.invitePending, pills:[...document.querySelectorAll('#friends .row')].map(r=>r.querySelector('.pill').className), cta:document.querySelector('#friends .bcta').classList.contains('off'), scroll:document.querySelector('#friends .scroll').scrollTop})")
    before = snap()
    click('#friends .row[data-k="otis"] .pill'); out["B_invite_opens_sheet"] = int(pg.evaluate("()=>SCENARIO.overlay==='startpoint' && !SCENARIO.invitePending && document.getElementById('friends').classList.contains('open') && document.elementFromPoint(196,150).closest('#friends,#startpoint')!==null"))   # over friend selection, nothing pending
    sp = pg.evaluate("()=>{const s=document.getElementById('startpoint'); const sh=s.querySelector('.sheet').getBoundingClientRect(); const kids=[...s.querySelector('.sheet').children].map(c=>c.className.split(' ')[0]); const d=s.querySelector('.dismiss').getBoundingClientRect(); return {open:s.classList.contains('open'), sheetBottom:sh.bottom, sheetH:sh.height, order:kids.join(','), dismissNearBottom:sh.bottom-d.bottom, sel:s.querySelector('.opt.sel').dataset.k, opt2:s.querySelector('.opt[data-k=\"shared\"] span').innerText.replace('\\n',' ')}}")
    out["sheet_bottom_packed"] = int(sp["open"] and abs(sp["sheetBottom"] - 852) < 0.5 and sp["sheetH"] < 420 and sp["dismissNearBottom"] < 40); out["sheet_order"] = int(sp["order"] == "handle,st,opt,opt,bcta,dismiss")
    out["shared_preselected_named"] = int(sp["sel"] == "shared" and sp["opt2"] == "Pick up together Section 2, Unit 1")
    click('#startpoint .opt[data-k="beginning"]'); out["option1_fork"] = int(pg.evaluate("()=>document.getElementById('oos').classList.contains('open') && document.querySelector('#oos .msg').textContent==='Not in this demo — tap Pick up together' && document.querySelector('#startpoint .opt.sel').dataset.k==='shared'")); pg.mouse.click(200, 90); pg.wait_for_timeout(100)
    click("#startpoint .dismiss"); out["cancel_unchanged"] = int(snap() == before and pg.evaluate("()=>SCENARIO.overlay==='friends'"))   # CANCEL leaves the screen exactly as it was
    click('#friends .row[data-k="otis"] .pill'); click("#startpoint .bcta")
    out["send_commits_pending"] = int(pg.evaluate("()=>SCENARIO.overlay==='friends' && SCENARIO.invitePending==='otis' && document.querySelector('#friends .row[data-k=\"otis\"] .pill').textContent==='Pending' && [...document.querySelectorAll('#friends .row:not([data-k=\"otis\"]) .pill')].every(p=>p.classList.contains('off')) && !document.querySelector('#friends .bcta').classList.contains('off')"))
    click("#friends .bcta"); out["send_to_path"] = int(pg.evaluate("()=>CURRENT==='B2' && SCENARIO.overlay==='none'"))
    pg.wait_for_timeout(2300); out["B_acceptance_arrives"] = int(pg.evaluate("()=>SCENARIO.overlay==='accepted' && document.querySelector('#accepted .sub').textContent===\"You'll start at Section 2, Unit 1!\""))
    click("#accepted .bcta"); out["continue_to_B7"] = int(pg.evaluate("()=>CURRENT==='B7' && SCENARIO.pairStatus==='active' && SCENARIO.section===2"))
    return out


def type_cells(a, pg):
    """Every type-and-character combination in the three type cells
    (confirmation's QUIZ TYPE, waiting's QUIZ SENT, the answered split cell):
    the character never overlaps the label and both stay inside the row —
    the character is 44 in every stat cell (one size per context; the type
    tiles keep 80), labels wrap where they must (Jamie, Sept 15)."""
    q = """()=>{const s=document.querySelector('.quiz.open'); let bad=0, n=0; for (const v of s.querySelectorAll('.srow .v')) { if (getComputedStyle(v.closest('.srow')).display==='none') continue; const im=v.querySelector('img:not(.clock)'); const t=v.querySelector('.t'); if(!im) continue; n++;
      const ri=im.getBoundingClientRect(), rt=t.getBoundingClientRect(), rr=(v.closest('.half')||v.closest('.srow')).getBoundingClientRect(); const overlap=(ri.right>rt.left+0.5 && ri.left<rt.right && ri.bottom>rt.top+0.5 && ri.top<rt.bottom); if (overlap || rt.bottom>rr.bottom+0.5 || ri.bottom>rr.bottom+0.5 || rt.right>rr.right+0.5 || rt.left<rr.left-0.5 || Math.round(ri.height)!==44) bad++; } return [bad, n]}"""
    bad = 0; cells = 0
    for key in ("bea", "falstaff", "junior", "zari", "lily", "eddy", "lin", "vikram"):
        sec = 1 if key in ("bea", "falstaff", "junior", "zari") else 2
        for st in ("A13", "A15", "A16", "A18", "A20"):
            pg.evaluate(f"()=>applyPreset('{st}')"); pg.evaluate(f"()=>setState({{section:{sec}, unit:1, challengeType:'{key}'}})"); pg.wait_for_timeout(150)
            b, n = pg.evaluate(q); bad += b; cells += n
    return {"bad_cells": bad, "cells_checked": cells}


def centred(pg, sel):
    """The quiz base's rule: the body block centres between the title band
    (112) and the CTA (750.2; 828 without a CTA): top gap == bottom gap."""
    return pg.evaluate("""(sel)=>{const el=document.querySelector(sel); const b=el.querySelector('.qbody'); const kids=[...b.children].filter(c=>getComputedStyle(c).display!=='none');
      const r0=kids[0].getBoundingClientRect(), r1=kids[kids.length-1].getBoundingClientRect(), br=b.getBoundingClientRect();
      return {top:br.top, bottom:br.bottom, gapDiff:Math.abs((r0.top-br.top)-(br.bottom-r1.bottom)), inside:(r0.top>=br.top-0.5 && r1.bottom<=br.bottom+0.5)?1:0}}""", sel)


def re_rgb(s):
    import re
    if s.startswith("color("):
        m = [float(v) for v in re.findall(r"[0-9.]+", s.replace("display-p3", ""))][:3]; return [v * 255 for v in m]
    return [float(v) for v in re.findall(r"[0-9.]+", s)][:3]


def slot(a, pg):
    """The scroll-return control and the direction bubble as a mutually
    exclusive pair (Jamie, Sept 14): control iff the anchor node is off
    screen; bubble iff it is on screen and the marker is off. btn/bub from
    the DOM, plus the white ink in both slots so a hidden-by-CSS bubble that
    still paints, or vice versa, is caught by pixels."""
    w = (a >= 250).all(-1)
    d = pg.evaluate("()=>{const b=document.getElementById('bubble'), s=document.getElementById('scrollbtn'); return [s.style.display==='block'?1:0, b.style.display==='block'?1:0]}")
    return {"btn": d[0], "bub": d[1], "both": d[0] * d[1],
            "white": int(w[188 * DPR:245 * DPR, 321 * DPR:377 * DPR].sum() + w[685 * DPR:741 * DPR, 321 * DPR:377 * DPR].sum())}


def marker_mid_trophy(a, pg):
    """Marker on the trophy (A10, anchor scroll 139.05): left side, 50 degrees."""
    w = (a >= 250).all(-1)
    b = largest(w, 520, 580, 140, 205)   # above the trophy's own white glyph (top 582)
    return {"top": b[1], "h": b[3] - b[1], "bcx": (b[0] + b[2]) / 2} if b else {}


def marker_score6(a, pg):
    """Marker on the score 6 rosette (A10 after the score simulate; the path
    is at its maximum scroll, 319.8). The rosette is not a face ellipse, so
    the check is against its RENDERED INK: the pin's white against purple
    hue inside the node's box (the box excludes Otis's purple disc in the
    pin). viol = how far inside the 4px gap the nearest white pixel sits."""
    from scipy import ndimage
    w = (a >= 250).all(-1)
    b = largest(w, 530, 625, 190, 275)
    out = {}
    if not b: return out
    Y0, Y1, X0, X1 = (int(v * DPR) for v in (530, 625, 190, 275))
    lbl, n = ndimage.label(w[Y0:Y1, X0:X1]); k = 1 + np.argmax(ndimage.sum(w[Y0:Y1, X0:X1], lbl, range(1, n + 1)))
    ys, xs = np.where(lbl == k)
    ros = np.zeros(w.shape, bool); by0, by1, bx0, bx1 = (int(v * DPR) for v in (596, 711, 158, 237))
    reg = a[by0:by1, bx0:bx1]
    ros[by0:by1, bx0:bx1] = (reg[..., 2] > 170) & (reg[..., 2] - reg[..., 1] > 40) & (reg[..., 0] > 120)
    d = ndimage.distance_transform_edt(~ros)[ys + Y0, xs + X0].min() / DPR
    out.update(top=b[1], h=b[3] - b[1], bcx=(b[0] + b[2]) / 2, dist=float(d), viol=max(0.0, 4.0 - float(d)))
    return out


CHECKS = [
    # (screen, preset, setup JS, measure, {name: (target, tolerance)})
    ("A0 banner", "A0", None, banner(blue), {"width": (345.0, 2), "top": (111.4, 2)}),
    ("A1 banner", "A1", None, banner(green), {"width": (345.0, 2), "top": (111.4, 2)}),
    ("A1 trophy", "A1", "()=>{const s=document.getElementById('scroller'); s.scrollTop=s.scrollHeight}",
     trophy, {"cx": (196.7, 2), "cy": (741.1, 2)}),
    ("menu panel", "A0", "()=>{SCENARIO.overlay='courseMenu'; renderMenu()}",
     menu_panel, {"bottom": (613.6, 2)}),   # 548.3 + 65.30 LEARN WITH A FRIEND row
    ("feed plate", "A3", None, feed_plate, {"cx": (FP[0], 2), "cy": (FP[1], 2)}),
    # cx catches the row sliding: it is centred on the composite (Sept 11),
    # not spanning it, and nothing else would notice it moving.
    ("feed flag", "A3", None, feed_flag, {"w": (44.00, 2), "h": (34.00, 2), "cx": (301.33, 2),
      # the numeral is set EQUAL in height to the flag (Sept 11): its own
      # height needs a probe, or a numeral back at its old cap passes.
      "num_h": (33.33, 2), "num_cx": (340.17, 2)}),
    ("feed pair", "A3", None, feed_pair, {"cx": (FD[0], 2), "cy": (FD[1], 2), "w": (FD[2], 2), "corner": (0, 1.5)}),
    # Stage 2: partner marker on A7. First node, ringed, under the banner: the
    # ladder floors at 20 degrees and the pin shrinks to body 28 (r 14) to clear
    # the banner's 187.7 bottom by >= 4. h = intended 35 at that size.
    ("A7 marker", "A7", None, marker, {"top": (192.33, 2), "tip": (227.0, 2), "h": (34.67, 1), "w": (27.67, 2),
      "bcx": (243.83, 2), "rho": (1.10, 0.05), "dcx": (243.67, 2), "dw": (22.67, 2), "dxc": (0, 1)}),
    ("A7 taps", "A7", None, marker_taps, {"body_clv": (1, 0), "body_popup": (0, 0), "tail_popup": (1, 0), "tail_clv": (1, 0),
      "shoulder_popup": (1, 0)}),
    ("A1 no marker", "A1", None, marker_off, {"white": (0, 0)}),
    ("A7 marker node1", "A7", "()=>setState({partnerProgress:1})", marker_mid,
      {"top": (255.33, 2), "h": (39.33, 1), "bcx": (125.5, 2), "viol_n0": (0, 0.005), "viol_n2": (0, 0.005), "viol_own": (0, 0.005)}),
    # Direction bubble. B7 after REVIEW ALL: the anchor moves to the last
    # review node (S2U2 kanji) and Otis, at the extra unit's second star, is
    # ABOVE the viewport: top-right, point up, avatar upright, return control
    # hidden. After COMPLETE LESSON the anchor is Jamie's own node: still up.
    # The down case (Otis below, first-node anchor) is forced by state so the
    # direction logic is exercised both ways.
    ("B7 bubble up after REVIEW ALL", "B7", "()=>{tapNode('node0'); POPUP.firstReviewPopup.action(); tapNode('node15'); POPUP.lastReviewPopup.action()}", bubble((188, 245, 321, 377), "up"),
      {"btn": (0, 0), "top": (195.7, 1.5), "bot": (235.7, 1.5), "bcx": (349.0, 1.5), "dcx": (349.0, 1.5), "dw": (26.0, 2), "dxc": (0, 1), "upright": (1, 0)}),
    ("B7 bubble up after COMPLETE LESSON", "B7", "()=>{simulateReviewAll(); simulateCompleteLesson()}", bubble((188, 245, 321, 377), "up"),
      {"btn": (0, 0), "top": (195.7, 1.5), "bot": (235.7, 1.5), "bcx": (349.0, 1.5), "dcx": (349.0, 1.5), "dw": (26.0, 2), "dxc": (0, 1), "upright": (1, 0)}),
    ("B7 bubble down (Otis below, forced)", "B7", "()=>setState({partnerProgress:11})", bubble((685, 741, 321, 377), "down"),
      {"btn": (0, 0), "top": (693.0, 1.5), "bot": (733.0, 1.5), "bcx": (349.0, 1.5), "dcx": (349.0, 1.5), "dw": (26.0, 2), "dxc": (0, 1), "upright": (1, 0)}),
    ("B7 REVIEW keeps Otis on node 0", "B7", "()=>{tapNode('node0'); POPUP.firstReviewPopup.action()}", marker, {"top": (192.33, 2), "h": (34.67, 1)}),
    ("B7 no bubble at rest", "B7", None, bubble_off, {"white_top": (0, 0), "white_bot": (0, 0)}),
    # The pair. B7 with Otis below (node 11) from the first-node anchor: a
    # small scroll keeps the bubble and no control; scrolling the anchor node
    # (247) off the top swaps to the control. A7: the +-2px rule would have
    # shown the control at 40; the anchor-node rule does not until ~60.
    ("B7 small scroll keeps the bubble", "B7", "()=>{setState({partnerProgress:11}); document.getElementById('scroller').scrollTop=30}", slot, {"btn": (0, 0), "bub": (1, 0), "both": (0, 0)}),
    ("B7 far scroll swaps to the control", "B7", "()=>{setState({partnerProgress:11}); document.getElementById('scroller').scrollTop=300}", slot, {"btn": (1, 0), "bub": (0, 0), "both": (0, 0), "white": (0, 0)}),
    # marker-off is the marked NODE's centre (deviation from registry 284): at
    # 40px node 0 is still on screen, so no bubble either — the invisible band
    ("A7 no control while the anchor node shows", "A7", "()=>{document.getElementById('scroller').scrollTop=40}", slot, {"btn": (0, 0), "bub": (0, 0), "both": (0, 0)}),
    ("OOS forks", "A0", None, forks, {"friendA": (1, 0), "friendA_placed": (1, 0), "friendA_dismiss": (1, 0), "addA_works": (1, 0),
      "continue_inactive_silent": (1, 0), "continue": (1, 0), "continue_placed": (1, 0), "continue_dismiss": (1, 0),
      "addB": (1, 0), "addB_placed": (1, 0), "legendary": (1, 0), "legendary_placed": (1, 0), "review_uncovered": (1, 0),
      "legendary_dismiss": (1, 0), "review_works": (1, 0), "no_scrim": (1, 0), "fill_raised": (1, 0), "frame_white": (1, 0), "shadow": (1, 0)}),
    ("score 6 label white", "A10", "()=>{SCENARIO.overlay='score6Popup'; renderPopup()}", score6_label, {"label_lum": (255, 8)}),
    ("A7 control once the anchor node is off", "A7", "()=>{document.getElementById('scroller').scrollTop=100}", slot, {"btn": (1, 0), "bub": (0, 0), "both": (0, 0), "white": (0, 0)}),
    # Challenge tile: locked greys on A7 (band luma ~ locked face 62, card ~ glyph grey 96),
    # pink Send on A10 and B7, absent when unpaired or off the Japanese path.
    ("A7 tile locked", "A7", None, ctile, {"on": (1, 0), "left": (23.75, 0.5), "top": (282.2, 0.5), "gap": (15, 1), "label_ok": (1, 0),
      "band_lum": (63, 6), "card_lum": (96, 6), "pop_open": (1, 0), "pop_top": (360.9, 1), "caret_dx": (0, 1), "pop_copy": (1, 0), "pop_closes": (1, 0), "typesel": (0, 0)}),
    ("A10 tile send", "A10", None, ctile, {"on": (1, 0), "top": (282.2, 0.5), "label_ok": (1, 0), "band_pink": (1, 0), "card_lum": (255, 3), "pop_open": (0, 0), "typesel": (1, 0)}),
    ("B7 tile send", "B7", None, ctile, {"on": (1, 0), "label_ok": (1, 0), "band_pink": (1, 0), "typesel": (1, 0)}),
    ("A1 no tile", "A1", None, ctile, {"on": (0, 0)}),
    # Type selection (A11) and the question screens (A12) — §9, §10
    ("A11 type selection", "A11", None, typesel, {"open": (1, 0), "n_tiles": (4, 0), "tag_ok": (1, 0), "title_ok": (1, 0), "title_cy": (87, 1.5), "bg_tile_pink": (1, 0),
      "fan_of_five": (1, 0), "x_white": (1, 0), "body_top": (112, 0.5), "body_bottom": (750.2, 0.5), "centred": (1, 0), "labels_unique": (1, 0), "tile_translucent": (1, 0), "tile_no_dark": (1, 0), "no_outline": (1, 0), "tag_two_lines": (1, 0), "chars_equal_80": (1, 0), "label_areas_equal": (1, 0),
      "t0_l": (41, 0.5), "t0_w": (150, 0.5), "t0_h": (150, 0.5), "gap_x": (11, 0.5), "gap_y": (11, 0.5),
      "ready_before": (0, 0), "ready_after": (1, 0), "one_selected": (1, 0), "cta_white": (1, 0), "cta_l": (16.7, 0.5), "cta_t": (750.2, 0.5), "cta_w": (357.4, 0.5), "to_question": (1, 0)}),
    ("B7 type selection cards", "B7", "()=>openTypeSelection()", typesel, {"open": (1, 0), "n_tiles": (4, 0), "labels_unique": (1, 0), "to_question": (1, 0)}),
    ("A12 question bea (speak)", "A12", None, question("speak"), {"open": (1, 0), "cta_mode_ok": (1, 0), "n_segs": (5, 0), "bar_l": (62.6, 0.5), "bar_r": (299.1, 0.5),
      "bar_t": (87.45, 0.5), "bar_h": (15.7, 0.5), "on_count": (0, 0), "seg1_grey": (1, 0), "gap_is_page": (1, 0),
      "lab_ok": (1, 0), "lab_top": (753.1, 1), "lab_white": (1, 0), "energy_25": (1, 0), "energy_l": (346.2, 1), "energy_cy": (96.5, 1.5), "infinity_gone": (1, 0), "to_confirm": (1, 0), "energy_after": (20, 0)}),
    ("A12 question junior (kb)", "A12", "()=>setState({challengeType:'junior'})", question("kb"), {"open": (1, 0), "cta_mode_ok": (1, 0), "on_count": (0, 0), "gap_is_page": (1, 0), "lab_top": (732, 1), "plate_h": (49.1, 0.5), "plate_left": (75.7, 0.5), "lab_white": (1, 0), "infinity_gone": (1, 0), "to_confirm": (1, 0)}),
    ("A12 question zari (plate)", "A12", "()=>setState({challengeType:'zari'})", question("plate"), {"open": (1, 0), "cta_mode_ok": (1, 0), "gap_is_page": (1, 0), "lab_top": (753.1, 1), "lab_white": (1, 0), "infinity_gone": (1, 0), "to_confirm": (1, 0)}),
    ("A12-A17 sender loop", "A12", None, sender_loop, {"confirm_open": (1, 0), "hero_4": (1, 0), "hero_cx": (196.5, 1), "hero_dia": (186.7, 0.5), "confirm_centred": (1, 0),
      "line_ok": (1, 0), "cta_send": (1, 0), "energy_20": (1, 0), "no_percent": (1, 0), "no_x": (1, 0), "pair_w": (240, 1), "pair_cx": (196.5, 1), "row_type": (1, 0), "no_combo_row": (1, 0), "num_pink": (1, 0), "of_5": (1, 0), "label_quiz_type": (1, 0), "pair_ring_gap": (16, 1), "pair_clean": (1, 0),
      "sent_to_path": (1, 0), "tile_waiting": (1, 0),
      "status_open": (1, 0), "waiting_state": (1, 0), "no_countdown": (1, 0), "no_report": (1, 0), "no_nudge": (1, 0), "head_waiting": (1, 0), "sim1": (1, 0), "sim_tag": (1, 0), "type_row": (1, 0), "no_card": (1, 0), "sim_on_colour": (1, 0),
      "label_quiz_sent": (1, 0), "sub_two_lines": (1, 0), "no_combo_waiting": (1, 0), "waiting_centred": (1, 0), "head_33": (1, 0), "head_above_pair": (1, 0), "pair_240": (1, 0), "sim_one_line": (1, 0), "sim_pinned": (1, 0), "sub_21": (1, 0), "x_white": (1, 0), "x_paints": (1, 0), "x_cx": (28.07, 1), "x_cy": (86.7, 1), "x_w": (23, 1.5),
      "answered_state": (1, 0), "report_shown": (1, 0), "five_rows": (1, 0), "row_states_ok": (1, 0), "combo_1": (1, 0), "ok_text_pink": (1, 0), "miss_text_white": (1, 0), "no_green": (1, 0),
      "nudge_on": (1, 0), "nudge_white_cta": (1, 0), "nudge_above_control": (1, 0), "nudge_in_cta_slot": (1, 0), "sim_pinned_answered": (1, 0), "sim2": (1, 0), "head_answered": (1, 0), "sim_fits": (1, 0), "sub_turn_state": (1, 0), "one_margin": (1, 0), "stats_below_rows": (1, 0), "split_two_halves": (1, 0), "answered_centred": (1, 0), "nudge_fork": (1, 0), "your_turn": (1, 0)}),
    ("A12 forks and settings", "A12", None, question_forks, {"mic_popup": (1, 0), "mic_card_clear_of_plate": (1, 0), "mic_card_gap": (4, 1), "gear_opens_sheet": (1, 0), "done_popup": (1, 0), "done_card_clear_of_end": (1, 0), "end_to_path": (1, 0), "tile_still_send": (1, 0)}),
    ("A1 no control at rest", "A1", None, slot, {"btn": (0, 0), "bub": (0, 0)}),
    ("assets load", "A1", None, assets_load, {"missing": (0, 0), "unloadable": (0, 0)}),
    ("walk A, 24 beats", "A0", None, walk_A, {k: (1, 0) for k in ["b2_course_menu", "b3_course_selection", "b4_friends", "b5_solo_path", "b6_accepted", "b7_layer", "b8_clv", "b9_node_popup", "b10_path_send", "b11_typesel", "b12_question", "b13_confirm", "b14_path_waiting", "b15_status_waiting", "b16_answered", "b17_path_answer", "b18_state3", "b19_his_question", "b20_state4", "b21_path_send", "b22_score6_popup", "b23_milestone", "b23b_path", "b24_feed"]}),
    ("walk B, 13 beats", "B2", None, walk_B, {k: (1, 0) for k in ["b1_solo_no_rings", "b2_course_menu", "b3_friends", "b3_otis_score_shown", "b4_sheet", "b4b_pending", "b5_her_path", "b6_accepted", "b7_rings_and_marker", "b8_clv", "b9_first_ringed_popup", "b10_review_marker_stays", "b11_last_ringed_popup", "b12_review_all", "b13_her_node_popup", "b13_complete_lesson"]}),
    ("intro popup + picker", "A1", None, intro_popup, {"open_on_load": (1, 0), "frame_white_dashed": (1, 0), "tag": (1, 0), "title_green": (1, 0), "sub_ok": (1, 0), "card_313": (1, 0), "lines_held": (1, 0), "tag_on_corner": (1, 0), "tag_clears_title_ink": (1, 0), "rows_verbatim_one_line": (1, 0), "pill_start_here_green": (1, 0), "note_quiet": (1, 0), "no_pink_in_chrome": (1, 0), "pick_B_beat1": (1, 0), "bar_green": (1, 0), "selected_green": (1, 0), "others_neutral": (1, 0), "picker_collapsed": (1, 0), "reset_to_beat1": (1, 0), "reset_B_to_B2": (1, 0)}),
    ("feed to path", "A3", None, feed_to_path, {"feed_tab_live": (1, 0), "to_feed": (1, 0), "back_to_paired_path": (1, 0), "A3_back_paired": (1, 0), "solo_feed_not_live": (1, 0), "B_feed_never_live": (1, 0), "no_score_under_5": (1, 0)}),
    ("A16-A20 receiver loop", "A16", None, receiver_loop, {"sent_back": (1, 0), "type_differs": (1, 0), "state3": (1, 0), "s3_no_rows": (1, 0), "s3_no_control": (1, 0), "s3_countdown": (1, 0), "s3_clock": (1, 0),
      "s3_answer_cta": (1, 0), "s3_centred": (1, 0), "answer_to_question": (1, 0), "energy_before": (20, 0), "state4": (1, 0), "energy_15": (1, 0), "combo_2": (1, 0), "s4_rows": (1, 0), "s4_footers": (1, 0),
      "s4_centred": (1, 0), "notnow_to_path": (1, 0), "sendback_to_typesel": (1, 0)}),
    ("type cells, all eight", "A13", None, type_cells, {"bad_cells": (0, 0), "cells_checked": (40, 0)}),
    ("A4-A6 / B3-B5 invite flow", "A0", None, invite_flow, {"friends_open": (1, 0), "friends_dark": (1, 0), "title_ok": (1, 0), "course_flag": (1, 0), "no_counter": (1, 0), "list_above_options": (1, 0), "cta_off_before": (1, 0),
      "rows_5": (1, 0), "row_h": (74.7, 0.5), "A_order": (1, 0), "A_otis_italian_5_no_japanese": (1, 0), "A_sabrina_spanish_8": (1, 0), "A_non_japanese_first": (1, 0), "A_japanese_ascending": (1, 0), "flags_not_words": (1, 0), "pills_equal": (1, 0), "paired_greyed_last": (1, 0), "finding_css": (1, 0), "contacts_no_fill": (1, 0),
      "fork_below_never_covers_otis": (1, 0), "pending_state": (1, 0), "A_continue_to_path": (1, 0), "A_acceptance_arrives": (1, 0), "accepted_A_copy": (1, 0), "sub_below_pair": (1, 0), "accepted_order": (1, 0),
      "accepted_dark": (1, 0), "pair_264": (1, 0), "no_x_no_sim": (1, 0), "cta_continue": (1, 0), "continue_to_A7": (1, 0),
      "B_from_menu": (1, 0), "B_otis_first_at_10": (1, 0), "B_sorted": (1, 0), "B_invite_opens_sheet": (1, 0), "sheet_bottom_packed": (1, 0), "sheet_order": (1, 0), "shared_preselected_named": (1, 0),
      "option1_fork": (1, 0), "cancel_unchanged": (1, 0), "send_commits_pending": (1, 0), "send_to_path": (1, 0), "B_acceptance_arrives": (1, 0), "continue_to_B7": (1, 0)}),
    ("A21 milestone", "A21", None, milestone, {"open": (1, 0), "centred": (1, 0), "duo_168": (1, 0), "six": (1, 0), "flag_w": (74.7, 0.5), "order_ok": (1, 0), "line_ok": (1, 0), "cta_top": (683.4, 0.5), "blue": (1, 0), "pair_w": (240, 1),
      "share_fork": (1, 0), "continue_to_path": (1, 0), "fires_from_score6": (1, 0)}),
    ("A8 co-learning view", "A8", None, colearn, {"open": (1, 0), "hdr_pink": (1, 0), "hdr_full_bleed": (1, 0), "page_dark": (1, 0), "x_own_row_above": (1, 0), "x_paints": (1, 0), "no_header_text": (1, 0), "buddy_quest": (1, 0), "countdown": (1, 0),
      "figures_right": (1, 0), "hero_in_header": (1, 0), "bq_28_cd_15": (1, 0), "block_centred": (1, 0), "band_figures_card_unmoved": (1, 0), "clock_13": (1, 0), "counter_readable": (1, 0), "headers": (1, 0), "one_margin": (1, 0), "turn_locked_row": (1, 0), "turn_locked_inert": (1, 0), "locked_copy_one_line": (1, 0), "days_1_icon": (1, 0), "ov_one_line": (1, 0), "fig_ink_on_card_edge": (1, 0), "fig_below_x_row": (1, 0), "card_over_bottom_edge_only": (1, 0), "hdr_compact": (1, 0), "sections_double": (1, 0), "calendar_icon": (1, 0), "day_singular": (1, 0), "combo_shown_at_zero": (1, 0),
      "four_badges": (1, 0), "scores": (1, 0), "avatar_44": (1, 0), "no_sim": (1, 0), "end_small_last": (1, 0), "turn_to_state3": (1, 0), "turn_to_waiting": (1, 0), "turn_to_typesel": (1, 0),
      "combo_tracks": (1, 0), "month_capped_19": (1, 0), "B_scores_10_10": (1, 0), "marker_opens": (1, 0), "x_closes": (1, 0)}),
    ("feed after status", "A3", None, feed_after_status, {"overlays_open": (0, 0), "pair_l": (269.13, 1), "pair_t": (186.58, 1), "pair_w": (92.07, 1), "ringed_source": (1, 0)}),
    ("A2 no control at rest", "A2", None, slot, {"btn": (0, 0), "bub": (0, 0)}),
    ("B2 no control at rest", "B2", None, slot, {"btn": (0, 0), "bub": (0, 0)}),
    # A10 = A2 + the layer: Otis on the trophy; after the score 6 simulate he is
    # on the rosette, measured against its rendered purple ink (one-sided).
    ("A10 marker trophy", "A10", None, marker_mid_trophy, {"top": (534.8, 2), "h": (39.33, 1), "bcx": (172.6, 2)}),
    # the transition itself: SIMULATE COMPLETE UNIT from A7 must land on A10
    # with the layer intact, not on A2 (which predates it)
    ("A7 COMPLETE UNIT keeps the layer", "A7", "()=>{SCENARIO.overlay='firstNodePopup'; renderPopup(); POPUP.firstNodePopup.action()}", marker_mid_trophy,
      {"top": (534.8, 2), "h": (39.33, 1), "bcx": (172.6, 2)}),
    ("A10 marker score6", "A10", "()=>{SCENARIO.overlay='score6Popup'; renderPopup(); POPUP.score6Popup.action(); setState({overlay:'none'})}", marker_score6,
      {"top": (557.0, 2), "h": (39.33, 1), "bcx": (225.17, 2), "viol": (0, 0.005)}),
    ("B1 banner", "B1", None, banner(blue), {"width": (345.0, 2), "top": (111.4, 2)}),
    ("B1 extra banner", "B1", "()=>{document.getElementById('scroller').scrollTop=927}", banner(pale),
     {"width": (345.0, 2), "top": (111.4, 2)}),
    ("B1 node 0", "B1", None, node_at(196.92, 246.98), {"cx": (196.9, 2), "cy": (247.0, 2)}),
    ("B2 star", "B2", None, b2_star, {"cx": (126.7, 2), "cy": (409.0, 2)}),   # ref disc centre 409.0
    # 3b course selection (A0 -> Add). Targets read off the verified render.
    ("CS open", "A0", "()=>openCourseSelection()", cs_open,
     {"disc_cx": (CSO[0], 2), "disc_cy": (CSO[1], 2), "pill_top": (CSO[2], 2), "pill_bot": (CSO[3], 2),
      "row_blue": (23, 400)}),
    ("CS selected", "A0", "()=>{openCourseSelection(); selectCourse('japanese')}", cs_selected,
     {"cont_top": (CSS_[0], 2), "cont_right": (CSS_[1], 2), "ring_top": (CSS_[2], 2), "ring_bot": (CSS_[3], 2),
      "row_blue": (22741, 3000)}),
]


def run():
    ok = True
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 393, "height": 852}, device_scale_factor=DPR)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        # Verification-only: where Google Fonts is unreachable, serve Nunito from
        # a local file (NUNITO_TTF=/path/to/Nunito[wght].ttf). Text-dependent
        # checks — the out-of-scope card's placement is one — need the real
        # font; a fallback font reflows the string. No-op where fonts load.
        ttf = os.environ.get("NUNITO_TTF")
        if ttf and os.path.exists(ttf):
            css = f"@font-face{{font-family:'Nunito';src:url('file://{ttf}') format('truetype');font-weight:200 1000;}}"
            pg.route("**/fonts.googleapis.com/**", lambda r: r.fulfill(status=200, content_type="text/css", body=css))
        pg.goto(URL)
        pg.wait_for_timeout(500); pg.evaluate("()=>document.getElementById('intro')?.classList.remove('open')")   # the intro popup opens on load; checks run past it (its own check reloads)
        pg.wait_for_timeout(600)
        for name, preset, js, measure, want in CHECKS:
            pg.evaluate(f'()=>applyPreset("{preset}")')
            if js:
                pg.evaluate(js)
            pg.wait_for_timeout(250)
            got = measure(shot(pg), pg)
            for k, (t, tol) in want.items():
                v = got.get(k)
                good = v is not None and abs(v - t) <= tol
                ok &= good
                print(f"{'PASS' if good else 'FAIL'}  {name:11s} {k:6s} "
                      f"{'—' if v is None else f'{v:7.2f}'}  (want {t} ± {tol})")
        pg.unroute("**/fonts.googleapis.com/**")   # a live route handler keeps close() from returning
        b.close()
    if errs:
        ok = False
        print("FAIL  page errors:", errs)
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
