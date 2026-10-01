"""Draw the "Climbing the Nines" banner: a success-rate landscape as contour lines, with a
stochastic hill-climbing path from a pretrained policy (~50%) up past the 99% contour.

    uv run python tools/make_banner.py     # writes media/banner-light.svg and media/banner-dark.svg

The landscape and the path are the same deterministic construction as the hero of the explainer
site, so the README banner and the site match. The picture is illustrative: the axes are not
real policy parameters.
"""
from __future__ import annotations

import math
from pathlib import Path

import contourpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 360
A = W / H  # the landscape lives on [0, A] x [0, 1] so contours keep their shape at any aspect ratio
SIZES = {"banner": (1280, 360), "social-preview": (1280, 640)}  # README banner; GitHub/Twitter link card

THEMES = {
    "light": dict(bg="#F1F4EE", contour="#A7B4A2", accent="#1C58C0", ink="#17211C", muted="#48564E"),
    "dark": dict(bg="#0E1411", contour="#33433A", accent="#86A8FF", ink="#E2E9E4", muted="#A5B2AA"),
}
LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
LABELS = {0.5: "50%", 0.9: "90%", 0.99: "99%"}
FONT = "'Bricolage Grotesque', 'Avenir Next', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', SFMono-Regular, Menlo, Consolas, monospace"


def field(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    """Success rate as a function of two made-up policy parameters: one broad peak near 99.5%
    plus two foothills, so a greedy climber has something to get around."""
    x = x * (1280 / 360) / A  # same landscape on every canvas: defined in banner coordinates, stretched to fit
    cx, cy = 0.79 * 1280 / 360, 0.42
    r2 = ((x - cx) / 1.25) ** 2 + ((y - cy) / 0.8) ** 2
    main = 0.995 / (1 + r2 / 3.1)
    b = 0.32 * np.exp(-((x - 0.16 * 1280 / 360) ** 2 / 0.35 + (y - 0.28) ** 2 / 0.05))
    b2 = 0.22 * np.exp(-((x - 0.47 * 1280 / 360) ** 2 / 0.22 + (y - 0.86) ** 2 / 0.04))
    return np.minimum(0.999, main + (b + b2) * (1 - main))


def climb() -> list[tuple[float, float]]:
    """Stochastic hill climbing: try 6 random steps, keep the best one that improves, repeat."""
    seed = 7

    def rnd() -> float:
        nonlocal seed
        seed = (seed * 16807) % 2147483647
        return seed / 2147483647

    # start where the landscape reads ~50%, like the course's base policy (base_v1 scores 50.4%)
    y, step = 0.74, 0.055
    x = min(np.linspace(0.05 * A, 0.6 * A, 400), key=lambda v: abs(float(field(v, y)) - 0.50))
    path = [(x, y)]
    for _ in range(400):
        f0 = float(field(x, y))
        if f0 > 0.992:
            break
        best, bf = None, f0
        for _ in range(6):
            ang = rnd() * math.pi * 2
            nx, ny = x + math.cos(ang) * step, y + math.sin(ang) * step * 0.8
            if not 0.04 < ny < 0.96:
                continue
            fn = float(field(nx, ny))
            if fn > bf:
                bf, best = fn, (nx, ny)
        if best:
            x, y = best
            path.append(best)
    return path


def px(x: float, y: float) -> tuple[float, float]:
    return x / A * W, y * H


def svg(theme: str, size: str = "banner") -> str:
    global W, H, A
    W, H = SIZES[size]
    A = W / H
    big = size == "social-preview"
    c = THEMES[theme]
    xs, ys = np.linspace(0, A, 360), np.linspace(0, 1, 120)
    gx, gy = np.meshgrid(xs, ys)
    gen = contourpy.contour_generator(gx, gy, field(gx, gy))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-label="Climbing the Nines: a success-rate landscape drawn as contour lines, with a '
           f'dashed hill-climbing path from a pretrained policy at about 50% up past the 99% contour.">',
           f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>']
    label_at = {}
    for level in LEVELS:
        nines = level >= 0.9
        for line in gen.lines(level):
            pts = " ".join(f"{px(x, y)[0]:.1f},{px(x, y)[1]:.1f}" for x, y in line)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{c["accent"] if nines else c["contour"]}" '
                       f'stroke-width="{1.6 if nines else 1.1}" stroke-opacity="{0.9 if nines else 1}"/>')
            if level in LABELS:  # label each highlighted contour on its left flank, below the title and the path
                flank = [p for p in line if p[0] < 0.79 * A and 0.5 < p[1] < 0.9]
                if flank:
                    p = min(flank, key=lambda q: abs(q[1] - 0.66))
                    if level not in label_at or abs(p[1] - 0.66) < abs(label_at[level][1] - 0.66):
                        label_at[level] = p
    for level, (x, y) in label_at.items():
        if level == 0.99:
            continue  # the summit marker sits right on the 99% ring; its label says it instead
        X, Y = px(x, y)
        color = c["accent"] if level >= 0.9 else c["muted"]
        out.append(f'<rect x="{X - 19:.1f}" y="{Y - 9:.1f}" width="38" height="18" fill="{c["bg"]}"/>')
        out.append(f'<text x="{X:.1f}" y="{Y + 4:.1f}" text-anchor="middle" font-family="{MONO}" '
                   f'font-size="12" font-weight="600" fill="{color}">{LABELS[level]}</text>')

    path = [px(x, y) for x, y in climb()]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in path)
    out.append(f'<polyline points="{d}" fill="none" stroke="{c["ink"]}" stroke-width="2" stroke-dasharray="5 5"/>')
    out += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{c["ink"]}"/>' for x, y in path[::3]]
    (sx, sy), (ex, ey) = path[0], path[-1]
    start_rate = round(100 * float(field(*climb()[0])))
    out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6.5" fill="{c["bg"]}" stroke="{c["ink"]}" stroke-width="2.4"/>')
    out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="7.5" fill="{c["accent"]}"/>')
    out.append(f'<text x="{sx + 13:.1f}" y="{sy + 20:.1f}" font-family="{FONT}" font-size="15" font-weight="600" '
               f'fill="{c["ink"]}" stroke="{c["bg"]}" stroke-width="5" paint-order="stroke">pretrained policy · {start_rate}%</text>')
    out.append(f'<text x="{ex + 14:.1f}" y="{ey + 26:.1f}" font-family="{FONT}" font-size="15" '
               f'font-weight="600" fill="{c["ink"]}" stroke="{c["bg"]}" stroke-width="5" paint-order="stroke">last mile · 99%+</text>')
    # title block, top left
    ty, ts = (110, 76) if big else (74, 46)
    out.append(f'<text x="48" y="{ty}" font-family="{FONT}" font-size="{ts}" font-weight="800" letter-spacing="-1.2" '
               f'fill="{c["ink"]}">Climbing the <tspan fill="{c["accent"]}">Nines</tspan></text>')
    out.append(f'<text x="51" y="{ty + (46 if big else 30)}" font-family="{FONT}" font-size="{26 if big else 17}" '
               f'fill="{c["muted"]}">Awesome Hill Climbing · last-mile improvement of robot policies</text>')
    out.append(f'<text x="{W - 16}" y="{H - 14}" text-anchor="end" font-family="{MONO}" font-size="11" '
               f'fill="{c["muted"]}">success-rate landscape · illustrative</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    for theme in THEMES:
        path = ROOT / "media" / f"banner-{theme}.svg"
        path.write_text(svg(theme), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")
    # The social preview must be a PNG for GitHub (Settings > Social preview). Write the SVG here and
    # convert it with: rsvg-convert -w 1280 media/social-preview.svg -o media/social-preview.png
    path = ROOT / "media" / "social-preview.svg"
    path.write_text(svg("light", "social-preview"), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
