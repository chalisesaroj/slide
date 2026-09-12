"""
lecture.py
==========
Asset builder for the lecture:

    "Signal Conditioning and Processing"
    (Signals, Wheatstone bridge, Noise, DSP, Fourier methods, Data presentation)

Usage
-----
    python lecture.py              # build every static figure (SVG) into ./figures/
    python lecture.py --manim      # additionally render the optional Manim animation

All static assets are written as VECTOR (.svg) files into ./figures/.
The optional animation is written as ./figures/fourier_synthesis.mp4.

This file only *generates* assets. It never renders the presentation.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INK = "#12263A"
BLUE = "#1F6FB2"
ORANGE = "#E07B39"
GREEN = "#2E8B57"
RED = "#C0392B"
GREY = "#7F8C8D"
LIGHT = "#EAF2F8"
SOFT = "#F5F9FC"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 17,
    "axes.titlesize": 21,
    "axes.labelsize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.grid": True,
    "grid.color": "#C9D6E2",
    "grid.linewidth": 0.9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.22,
    "lines.linewidth": 2.6,
})


# ===========================================================================
# Generic helpers
# ===========================================================================
def save_svg(fig, name: str) -> None:
    """Save a Matplotlib figure as a vector SVG inside figures/."""
    path = FIG_DIR / name
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"[ok] {path}")


def esc(text: str) -> str:
    """Escape text for inclusion in SVG markup."""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))


def svg_open(w: int, h: int) -> str:
    """Return the SVG header, arrow markers and white background."""
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" '
        f'font-family="DejaVu Sans, Arial, Helvetica, sans-serif">\n'
        '<defs>\n'
        f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>\n'
        f'<marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/></marker>\n'
        '</defs>\n'
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>\n'
    )
    return head


def svg_write(name: str, parts) -> None:
    """Write the collected SVG fragments to figures/<name>."""
    path = FIG_DIR / name
    path.write_text("".join(parts) + "</svg>\n", encoding="utf-8")
    print(f"[ok] {path}")


def s_text(x, y, text, size=24, fill=INK, anchor="middle", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'dominant-baseline="middle">{esc(text)}</text>\n')


def s_line(x1, y1, x2, y2, stroke=INK, sw=3.0):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"/>\n')


def s_arrow(x1, y1, x2, y2, stroke=INK, sw=3.5):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" '
            f'marker-end="url(#ah)"/>\n')


def s_rect(x, y, w, h, rx=12, fill=SOFT, stroke=BLUE, sw=3.0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n')


def s_block(x, y, w, h, title, sub="", fill=SOFT, stroke=BLUE,
            tsize=21, ssize=16):
    out = s_rect(x, y, w, h, rx=14, fill=fill, stroke=stroke, sw=3.0)
    cy = y + h / 2.0
    if sub:
        out += s_text(x + w / 2.0, cy - 20, title, size=tsize, weight="bold")
        out += s_text(x + w / 2.0, cy + 20, sub, size=ssize, fill="#41576B")
    else:
        out += s_text(x + w / 2.0, cy, title, size=tsize, weight="bold")
    return out


def arm_with_resistor(x1, y1, x2, y2, rw=120, rh=44,
                      fill="#FFFFFF", stroke=INK, sw=3.5):
    """Draw a bridge arm as a line carrying a rectangular resistor body."""
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    out = s_line(x1, y1, x2, y2, stroke=stroke, sw=sw)
    out += (f'<rect x="{cx - rw / 2:.1f}" y="{cy - rh / 2:.1f}" '
            f'width="{rw}" height="{rh}" rx="6" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" '
            f'transform="rotate({ang:.2f} {cx:.1f} {cy:.1f})"/>\n')
    return out


# ===========================================================================
# 2.1  Signals
# ===========================================================================
def fig_signal_types():
    """Analog / discrete-time / digital representation of the same signal."""
    t = np.linspace(0.0, 1.0, 2000)
    x = np.sin(2 * np.pi * 2 * t) + 0.35 * np.sin(2 * np.pi * 5 * t)

    ts = np.linspace(0.0, 1.0, 21)
    xs = np.sin(2 * np.pi * 2 * ts) + 0.35 * np.sin(2 * np.pi * 5 * ts)

    levels = np.linspace(-1.4, 1.4, 8)
    idx = np.argmin(np.abs(xs[:, None] - levels[None, :]), axis=1)
    xq = levels[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    axes[0].plot(t, x, color=BLUE, lw=2.8)
    axes[0].set_title("Analog signal\ncontinuous in time and amplitude")

    markerline, stemlines, _ = axes[1].stem(ts, xs, basefmt=" ")
    plt.setp(stemlines, color=GREY, linewidth=1.6)
    plt.setp(markerline, color=ORANGE, markersize=9,
             markeredgecolor=ORANGE)
    axes[1].set_title("Discrete-time signal\nsampled at instants $nT_s$")

    for lv in levels:
        axes[2].axhline(lv, color="#D5E1EC", lw=1.1, zorder=1)
    axes[2].step(ts, xq, where="post", color=ORANGE, lw=3.0, zorder=3)
    axes[2].plot(ts, xs, "o", color=INK, ms=5, zorder=4)
    axes[2].set_title("Digital signal\nsampled and quantised")

    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (V)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-1.9, 1.9)
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    save_svg(fig, "signal_types.svg")


def fig_sampling_quantization():
    """Sampling (time discretisation) and quantisation (amplitude discretisation)."""
    fs = 20.0
    f0 = 3.0

    t = np.linspace(0.0, 0.5, 2000)
    x = np.sin(2 * np.pi * f0 * t)

    n = np.arange(0, 11)
    ts = n / fs
    xs = np.sin(2 * np.pi * f0 * ts)

    t2 = np.linspace(0.0, 0.5, 400)
    x2 = np.sin(2 * np.pi * f0 * t2)
    levels = np.linspace(-1.4, 1.4, 8)
    xq = levels[np.argmin(np.abs(x2[:, None] - levels[None, :]), axis=1)]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))

    axes[0].plot(t, x, color=BLUE, lw=2.6, label="continuous $x(t)$")
    axes[0].vlines(ts, 0, xs, color=GREY, lw=1.4)
    axes[0].plot(ts, xs, "o", color=ORANGE, ms=9, label="samples $x[n]$")
    axes[0].annotate("", xy=(ts[1], -1.45), xytext=(ts[2], -1.45),
                     arrowprops=dict(arrowstyle="<->", color=INK, lw=2.2))
    axes[0].text((ts[1] + ts[2]) / 2.0, -1.62, "$T_s$", ha="center",
                 va="top", fontsize=20)
    axes[0].set_title("Step 1 — Sampling: discrete in time")
    axes[0].set_ylim(-2.0, 1.6)
    axes[0].legend(loc="upper right", frameon=False)

    for lv in levels:
        axes[1].axhline(lv, color="#D5E1EC", lw=1.1, zorder=1)
    axes[1].plot(t2, x2, color=BLUE, lw=2.2, alpha=0.85,
                 label="analog input $x(t)$")
    axes[1].step(t2, xq, where="post", color=ORANGE, lw=3.0,
                 label="quantised output")
    axes[1].annotate("", xy=(0.42, levels[3]), xytext=(0.42, levels[4]),
                     arrowprops=dict(arrowstyle="<->", color=RED, lw=2.4))
    axes[1].text(0.44, (levels[3] + levels[4]) / 2.0, r"$\Delta$",
                 color=RED, fontsize=22, va="center")
    axes[1].set_title("Step 2 — Quantisation: discrete in amplitude")
    axes[1].set_ylim(-2.0, 1.6)
    axes[1].legend(loc="upper right", frameon=False)

    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (V)")
        ax.set_xlim(-0.01, 0.51)
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    save_svg(fig, "sampling_quantization.svg")


def fig_aliasing():
    """Aliasing: identical samples from a 7 Hz and an apparent 3 Hz component."""
    f0 = 7.0
    t = np.linspace(0.0, 0.6, 3000)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))

    # --- adequate sampling -------------------------------------------------
    fs_a = 20.0
    na = np.arange(0, int(0.6 * fs_a) + 1)
    ta = na / fs_a
    xa = np.sin(2 * np.pi * f0 * ta)
    axes[0].plot(t, np.sin(2 * np.pi * f0 * t), color=BLUE, lw=2.6,
                 label="$f = 7$ Hz")
    axes[0].vlines(ta, 0, xa, color=GREY, lw=1.4)
    axes[0].plot(ta, xa, "o", color=ORANGE, ms=8, label="samples")
    axes[0].set_title("$f_s = 20$ Hz  >  $2f$   →  no aliasing")
    axes[0].legend(loc="upper right", frameon=False, ncol=2)

    # --- inadequate sampling ----------------------------------------------
    fs_b = 10.0
    nb = np.arange(0, int(0.6 * fs_b) + 1)
    tb = nb / fs_b
    xb = np.sin(2 * np.pi * f0 * tb)
    axes[1].plot(t, np.sin(2 * np.pi * f0 * t), color=BLUE, lw=2.6,
                 label="true 7 Hz")
    axes[1].plot(t, -np.sin(2 * np.pi * 3.0 * t), color=ORANGE, lw=2.4,
                 ls="--", label="apparent 3 Hz (alias)")
    axes[1].vlines(tb, 0, xb, color=GREY, lw=1.4)
    axes[1].plot(tb, xb, "o", color=INK, ms=8, label="samples")
    axes[1].set_title("$f_s = 10$ Hz  <  $2f$   →  aliasing")
    axes[1].legend(loc="upper right", frameon=False, ncol=1)

    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (V)")
        ax.set_xlim(-0.01, 0.61)
        ax.set_ylim(-1.55, 1.55)
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    save_svg(fig, "aliasing.svg")


def fig_acquisition_chain():
    """Analog → digital acquisition chain (hand-built SVG)."""
    W, H = 1700, 460
    parts = [svg_open(W, H)]

    blocks = [
        ("Physical Quantity", "strain · temperature"),
        ("Sensor", "converts to electrical"),
        ("Signal Conditioning", "amplify · filter"),
        ("ADC", "sample · quantise"),
        ("Digital Processing", "DSP · DFT / FFT"),
        ("Display / Storage", "plot · record · act"),
    ]

    bw, bh, gap = 240, 140, 40
    x0, y0 = 30, 190

    for i, (title, sub) in enumerate(blocks):
        x = x0 + i * (bw + gap)
        fill = SOFT if i < 3 else "#FDF1E7"
        stroke = BLUE if i < 3 else ORANGE
        parts.append(s_block(x, y0, bw, bh, title, sub,
                             fill=fill, stroke=stroke))

    for i in range(len(blocks) - 1):
        xa = x0 + i * (bw + gap) + bw
        xb = x0 + (i + 1) * (bw + gap)
        parts.append(s_arrow(xa + 5, y0 + bh / 2.0, xb - 5, y0 + bh / 2.0))

    xd = x0 + 3 * (bw + gap) - gap / 2.0
    parts.append(f'<line x1="{xd}" y1="120" x2="{xd}" y2="400" '
                 f'stroke="{GREY}" stroke-width="2.5" '
                 f'stroke-dasharray="10 8"/>\n')
    parts.append(s_text(430, 95, "ANALOG DOMAIN", size=22,
                        fill=BLUE, weight="bold"))
    parts.append(s_text(1270, 95, "DIGITAL DOMAIN", size=22,
                        fill=ORANGE, weight="bold"))

    svg_write("acquisition_chain.svg", parts)


# ===========================================================================
# 2.2  Wheatstone bridge
# ===========================================================================
def fig_wheatstone_bridge():
    """Wheatstone bridge schematic with a quarter-bridge active gauge."""
    W, H = 1150, 740
    parts = [svg_open(W, H)]

    T = (575, 140)
    L = (225, 370)
    R = (925, 370)
    B = (575, 600)

    parts.append(arm_with_resistor(*T, *L))   # R1  upper-left
    parts.append(arm_with_resistor(*T, *R))   # R2  upper-right
    parts.append(arm_with_resistor(*L, *B))   # R3  lower-left
    parts.append(arm_with_resistor(*R, *B))   # R4  lower-right (active)

    for (x, y) in (T, L, R, B):
        parts.append(f'<circle cx="{x}" cy="{y}" r="9" fill="{INK}"/>\n')

    # excitation terminals
    parts.append(s_line(T[0], T[1], T[0], 62))
    parts.append(f'<circle cx="{T[0]}" cy="52" r="10" fill="#ffffff" '
                 f'stroke="{INK}" stroke-width="3"/>\n')
    parts.append(s_text(602, 52, "+V\u209b", size=28, anchor="start",
                        weight="bold"))
    parts.append(s_line(B[0], B[1], B[0], 678))
    parts.append(f'<circle cx="{B[0]}" cy="688" r="10" fill="#ffffff" '
                 f'stroke="{INK}" stroke-width="3"/>\n')
    parts.append(s_text(602, 688, "\u2212V\u209b", size=28, anchor="start",
                        weight="bold"))

    # output branch with meter
    parts.append(s_line(L[0], L[1], 533, 370))
    parts.append(s_line(617, 370, R[0], R[1]))
    parts.append(f'<circle cx="575" cy="370" r="42" fill="#ffffff" '
                 f'stroke="{BLUE}" stroke-width="3.5"/>\n')
    parts.append(f'<text x="575" y="370" font-size="30" fill="{INK}" '
                 f'text-anchor="middle" dominant-baseline="middle" '
                 f'font-weight="bold">V<tspan font-size="20" dy="6">o</tspan>'
                 f'</text>\n')

    # component labels
    parts.append(s_text(357, 189, "R\u2081", size=28, weight="bold", fill=BLUE))
    parts.append(s_text(793, 189, "R\u2082", size=28, weight="bold", fill=BLUE))
    parts.append(s_text(357, 551, "R\u2083", size=28, weight="bold", fill=BLUE))
    parts.append(s_text(793, 551, "R\u2084 = R + \u0394R", size=25,
                        weight="bold", fill=ORANGE))

    svg_write("wheatstone_bridge.svg", parts)


def fig_wheatstone_response():
    """Quarter-bridge transfer characteristic and linearity error."""
    x = np.linspace(0.0, 0.05, 500)          # x = dR / R
    linear = 0.25 * x                        # Vo/Vs
    exact = 0.25 * x / (1.0 + 0.5 * x)       # Vo/Vs

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))

    axes[0].plot(x * 100, linear * 1000, color=ORANGE, lw=2.6, ls="--",
                 label=r"linear approx.  $V_o/V_s=\Delta R/4R$")
    axes[0].plot(x * 100, exact * 1000, color=BLUE, lw=3.0,
                 label=r"exact  $V_o/V_s=\frac{\Delta R/R}{4+2\Delta R/R}$")
    axes[0].set_xlabel(r"$\Delta R / R$  (%)")
    axes[0].set_ylabel(r"$V_o / V_s$  (mV/V)")
    axes[0].set_title("Quarter-bridge output vs resistance change")
    axes[0].legend(loc="upper left", frameon=False)

    err = (linear - exact) / exact * 100.0
    axes[1].plot(x * 100, err, color=RED, lw=3.0)
    axes[1].axhline(0, color=GREY, lw=1.2)
    axes[1].set_xlabel(r"$\Delta R / R$  (%)")
    axes[1].set_ylabel("linearity error (%)")
    axes[1].set_title("Deviation of the linear approximation")
    axes[1].annotate(f"{err[-1]:.1f} % at 5 %",
                     xy=(5.0, err[-1]), xytext=(3.0, err[-1] - 0.6),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.8),
                     fontsize=17, color=INK)

    for ax in axes:
        ax.grid(True, alpha=0.35)
        ax.set_xlim(0, 5.2)

    fig.tight_layout()
    save_svg(fig, "wheatstone_response.svg")


# ===========================================================================
# 2.3  Noise
# ===========================================================================
def fig_noise_spectrum():
    """Schematic power spectral density of the main noise mechanisms."""
    f = np.logspace(0, 4, 900)
    thermal = np.full_like(f, 1.0e-14)
    shot = np.full_like(f, 2.0e-15)
    flicker = 3.0e-12 / f

    fig, ax = plt.subplots(figsize=(12, 6.75))

    ax.loglog(f, flicker, color=GREEN, lw=3.0)
    ax.loglog(f, thermal, color=BLUE, lw=3.0)
    ax.loglog(f, shot, color=ORANGE, lw=3.0)

    ax.vlines([50, 150], 1e-16, [1e-11, 4e-12], color=RED, lw=2.6)
    ax.plot([50, 150], [1e-11, 4e-12], "v", color=RED, ms=10)

    ax.text(1.15e4, 3.0e-16, "flicker (1/f)", color=GREEN, fontsize=17,
            va="center", ha="left", clip_on=False, fontweight="bold")
    ax.text(1.15e4, 1.0e-14, "thermal (white)", color=BLUE, fontsize=17,
            va="center", ha="left", clip_on=False, fontweight="bold")
    ax.text(1.15e4, 2.0e-15, "shot (white)", color=ORANGE, fontsize=17,
            va="center", ha="left", clip_on=False, fontweight="bold")
    ax.text(58, 1.6e-11, "50 Hz mains\ninterference", color=RED, fontsize=15,
            va="bottom", ha="left")

    ax.set_xlim(1, 1e4)
    ax.set_ylim(1e-16, 5e-11)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("noise power density  (V$^2$/Hz, arbitrary)")
    ax.set_title("Typical noise power spectral density of a sensor channel")
    ax.grid(True, which="both", alpha=0.30)

    fig.tight_layout()
    save_svg(fig, "noise_spectrum.svg")


def fig_snr():
    """Time-domain and spectral illustration of the signal-to-noise ratio."""
    rng = np.random.default_rng(7)
    fs = 1000.0
    t = np.arange(0, 0.2, 1.0 / fs)
    amp = 1.0
    sigma = 0.1
    clean = amp * np.sin(2 * np.pi * 50 * t)
    noise = rng.normal(0.0, sigma, t.size)
    noisy = clean + noise

    snr_db = 20 * np.log10((amp / np.sqrt(2.0)) / sigma)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))

    axes[0].plot(t * 1000, clean, color=BLUE, lw=2.6, label="signal")
    axes[0].plot(t * 1000, noisy, color=GREY, lw=1.3, alpha=0.9,
                 label="signal + noise")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("amplitude (V)")
    axes[0].set_title(f"Waveform view   (SNR $\\approx$ {snr_db:.0f} dB)")
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[0].set_xlim(0, 60)

    N = t.size
    spec = np.abs(np.fft.rfft(noisy)) * 2.0 / N
    freq = np.fft.rfftfreq(N, 1.0 / fs)
    spec_clean = np.abs(np.fft.rfft(clean)) * 2.0 / N

    axes[1].semilogy(freq, spec + 1e-4, color=GREY, lw=1.6)
    axes[1].semilogy(freq, spec_clean + 1e-4, color=BLUE, lw=2.4)
    axes[1].annotate("signal peak\n(50 Hz)",
                     xy=(50, spec.max()), xytext=(140, spec.max() * 0.5),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.8),
                     fontsize=16, color=INK)
    axes[1].annotate("noise floor",
                     xy=(300, np.median(spec[150:])),
                     xytext=(330, np.median(spec[150:]) * 6),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.8),
                     fontsize=16, color=INK)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("amplitude (V)")
    axes[1].set_title("Spectral view: signal above the noise floor")
    axes[1].set_xlim(0, 500)

    for ax in axes:
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    save_svg(fig, "snr.svg")


# ===========================================================================
# 2.4  Digital signal processing
# ===========================================================================
def fig_dsp_cycle():
    """Four-stage DSP workflow drawn as a cycle (hand-built SVG)."""
    W, H = 1200, 760
    parts = [svg_open(W, H)]

    bw, bh = 470, 190
    boxes = [
        (80, 90, "1 · Acquire", "sensor \u2192 ADC, N samples"),
        (650, 90, "2 · Pre-process", "de-trend · window · scale"),
        (650, 480, "3 · Transform", "FIR / IIR filter · DFT / FFT"),
        (80, 480, "4 · Present & Act", "plot · store · alarm · control"),
    ]
    for (x, y, title, sub) in boxes:
        parts.append(s_block(x, y, bw, bh, title, sub,
                             tsize=26, ssize=17))

    parts.append(s_arrow(555, 185, 645, 185))
    parts.append(s_arrow(885, 285, 885, 475))
    parts.append(s_arrow(645, 575, 555, 575))
    parts.append(s_arrow(315, 475, 315, 285))

    svg_write("dsp_cycle.svg", parts)


def fig_dsp_filtering():
    """Demonstration of digital low-pass filtering on a noisy measurement."""
    rng = np.random.default_rng(3)
    fs = 200.0
    t = np.arange(0, 2.0, 1.0 / fs)

    wanted = 1.0 * np.sin(2 * np.pi * 3.0 * t)
    interference = 0.55 * np.sin(2 * np.pi * 40.0 * t)
    noise = rng.normal(0.0, 0.25, t.size)
    raw = wanted + interference + noise

    # 5-point moving average: null exactly at fs/5 = 40 Hz
    kernel = np.ones(5) / 5.0
    filtered = np.convolve(raw, kernel, mode="same")

    lo, hi = 12, -12

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), sharey=True)

    axes[0].plot(t[lo:hi], raw[lo:hi], color=GREY, lw=1.4)
    axes[0].set_title("Raw measurement\n3 Hz signal + 40 Hz interference + noise")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude (V)")

    axes[1].plot(t[lo:hi], raw[lo:hi], color="#D5DDE5", lw=1.2,
                 label="raw")
    axes[1].plot(t[lo:hi], filtered[lo:hi], color=BLUE, lw=3.0,
                 label="after 5-point moving average")
    axes[1].set_title("After digital low-pass filtering\n40 Hz component removed")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right", frameon=False)

    for ax in axes:
        ax.grid(True, alpha=0.35)
        ax.set_xlim(0.15, 1.85)
        ax.set_ylim(-2.2, 2.2)

    fig.tight_layout()
    save_svg(fig, "dsp_filtering.svg")


# ===========================================================================
# 2.5  Fourier transform
# ===========================================================================
def fig_dft_time_freq():
    """Time-domain record and its single-sided amplitude spectrum."""
    fs = 100.0
    N = 256
    t = np.arange(N) / fs

    x = (1.0 * np.sin(2 * np.pi * 5.0 * t)
         + 0.5 * np.sin(2 * np.pi * 12.0 * t)
         + 0.3 * np.sin(2 * np.pi * 20.0 * t))

    X = np.fft.rfft(x)
    freq = np.fft.rfftfreq(N, 1.0 / fs)
    amp = np.abs(X) * 2.0 / N

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    m = t <= 1.0
    axes[0].plot(t[m], x[m], color=BLUE, lw=2.4)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude (V)")
    axes[0].set_title("Time domain:  $x[n]$  ($N = 256$ samples, $f_s = 100$ Hz)")
    axes[0].set_xlim(0, 1.0)

    axes[1].stem(freq, amp, basefmt=" ")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("$|X[k]|$  (V)")
    axes[1].set_title("Frequency domain: single-sided amplitude spectrum")
    axes[1].set_xlim(0, 30)

    for fpk, apk in ((5.0, 1.0), (12.0, 0.5), (20.0, 0.3)):
        axes[1].annotate(f"{fpk:.0f} Hz", xy=(fpk, apk),
                         xytext=(fpk + 1.2, apk + 0.10),
                         fontsize=15, color=INK)

    for ax in axes:
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    save_svg(fig, "dft_time_freq.svg")


def fig_dft_vs_fft():
    """Operation count of the direct DFT versus the radix-2 FFT."""
    N = 2 ** np.arange(3, 14)
    dft = N.astype(float) ** 2
    fft = N.astype(float) * np.log2(N)

    fig, ax = plt.subplots(figsize=(12, 6.75))

    ax.loglog(N, dft, "o-", color=ORANGE, lw=2.8, ms=8,
              label=r"direct DFT  $\approx N^2$")
    ax.loglog(N, fft, "s-", color=BLUE, lw=2.8, ms=8,
              label=r"radix-2 FFT  $\approx N\log_2 N$")

    n0 = 1024
    ax.annotate("", xy=(n0, n0 * n0), xytext=(n0, n0 * np.log2(n0)),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2.4))
    ax.text(n0 * 1.15, (n0 * n0) ** 0.5 * (n0 * np.log2(n0)) ** 0.5,
            "  $\\approx$ 100$\\times$ fewer\n  operations at $N=1024$",
            color=RED, fontsize=16, va="center")

    ax.set_xlabel("transform length  $N$")
    ax.set_ylabel("number of operations")
    ax.set_title("Computational cost: DFT versus FFT")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    save_svg(fig, "dft_vs_fft.svg")


# ===========================================================================
# 2.6  Inverse Fourier transform
# ===========================================================================
def fig_inverse_ft():
    """Spectral truncation and reconstruction through the inverse DFT."""
    rng = np.random.default_rng(21)
    fs = 200.0
    N = 512
    t = np.arange(N) / fs

    x = (1.0 * np.sin(2 * np.pi * 5.0 * t)
         + 0.6 * np.sin(2 * np.pi * 15.0 * t)
         + 0.35 * np.sin(2 * np.pi * 45.0 * t)
         + rng.normal(0.0, 0.18, N))

    X = np.fft.fft(x)
    freq = np.fft.fftfreq(N, 1.0 / fs)

    keep = np.abs(X) > 0.30 * np.abs(X).max()
    X_clean = np.where(keep, X, 0.0)
    x_clean = np.fft.ifft(X_clean).real

    order = np.argsort(np.abs(X))[::-1]
    X_top = np.zeros_like(X)
    X_top[order[:8]] = X[order[:8]]
    x_top = np.fft.ifft(X_top).real

    half = slice(0, N // 2)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    axes[0].vlines(freq[half], 0, np.abs(X)[half] / N, color="#C9D6E2", lw=3.0,
                   label="removed coefficients")
    axes[0].vlines(freq[half][keep[half]], 0, np.abs(X)[half][keep[half]] / N,
                   color=BLUE, lw=3.0, label="retained coefficients")
    axes[0].set_xlim(0, 60)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("$|X[k]|/N$  (V)")
    axes[0].set_title("Step 1 — threshold the spectrum")
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(t, x, color="#C9D6E2", lw=1.5, label="original $x[n]$")
    axes[1].plot(t, x_clean, color=BLUE, lw=2.4,
                 label="IDFT of thresholded spectrum")
    axes[1].plot(t, x_top, color=ORANGE, lw=2.4, ls="--",
                 label="IDFT keeping only the 8 largest bins")
    axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("amplitude (V)")
    axes[1].set_title("Step 2 — inverse DFT returns to the time domain")
    axes[1].legend(loc="upper right", frameon=False, ncol=3)

    for ax in axes:
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    save_svg(fig, "inverse_ft.svg")


# ===========================================================================
# 2.7  Data presentation
# ===========================================================================
def fig_data_presentation():
    """Four standard ways of presenting the same measurement campaign."""
    rng = np.random.default_rng(11)

    hours = np.arange(0.0, 24.0, 0.25)
    temp = 22.0 + 6.0 * np.sin(2 * np.pi * (hours - 6.0) / 24.0) \
        + rng.normal(0.0, 0.6, hours.size)

    sensors = ["S1", "S2", "S3", "S4"]
    means = [21.8, 23.4, 22.1, 24.6]
    stds = [1.4, 1.1, 1.7, 1.3]

    humidity = 55.0 + 0.9 * (temp - 22.0) + rng.normal(0.0, 2.4, temp.size)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(hours, temp, color=BLUE, lw=2.4)
    axes[0, 0].set_xlabel("time (h)")
    axes[0, 0].set_ylabel("temperature (°C)")
    axes[0, 0].set_title("(a) Line chart — trend over time")
    axes[0, 0].set_xlim(0, 24)

    axes[0, 1].hist(temp, bins=16, color=BLUE, alpha=0.75,
                    edgecolor="white")
    axes[0, 1].set_xlabel("temperature (°C)")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title("(b) Histogram — distribution")

    axes[1, 0].bar(sensors, means, yerr=stds, capsize=8,
                   color=[BLUE, ORANGE, GREEN, RED], alpha=0.85,
                   edgecolor=INK)
    axes[1, 0].set_ylabel("mean temperature (°C)")
    axes[1, 0].set_title("(c) Bar chart with error bars — comparison")
    axes[1, 0].set_ylim(0, 30)

    axes[1, 1].scatter(temp, humidity, s=28, color=BLUE, alpha=0.65,
                       edgecolor="white")
    slope, intercept = np.polyfit(temp, humidity, 1)
    xs = np.linspace(temp.min(), temp.max(), 50)
    axes[1, 1].plot(xs, slope * xs + intercept, color=ORANGE, lw=3.0)
    axes[1, 1].set_xlabel("temperature (°C)")
    axes[1, 1].set_ylabel("relative humidity (%)")
    axes[1, 1].set_title("(d) Scatter plot with trend line — correlation")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    save_svg(fig, "data_presentation.svg")


# ===========================================================================
# Optional Manim animation
# ===========================================================================
try:                                                    # pragma: no cover
    from manim import (Scene, Axes, Create, Write, Text, MathTex,
                       UP, always_redraw, ValueTracker)
    from manim import BLUE as MANIM_BLUE, ORANGE as MANIM_ORANGE, GREY_B as MANIM_GREY_B
    MANIM_AVAILABLE = True
except Exception:                                       # pragma: no cover
    MANIM_AVAILABLE = False


if MANIM_AVAILABLE:                                     # pragma: no cover

    class FourierSynthesis(Scene):
        """Build a square wave by adding odd harmonics (Fourier synthesis)."""

        def construct(self):
            axes = Axes(
                x_range=[0, 2 * np.pi, np.pi / 2],
                y_range=[-1.6, 1.6, 0.5],
                x_length=10.5,
                y_length=4.6,
                tips=False,
                axis_config={"color": MANIM_GREY_B, "stroke_width": 3},
            )
            labels = axes.get_axis_labels(MathTex("t"), MathTex("x(t)"))

            title = Text(
                "Fourier synthesis: a square wave from odd harmonics",
                font_size=30,
            ).to_edge(UP, buff=0.5)

            def partial(x, terms):
                y = np.zeros_like(x)
                for i in range(max(1, int(terms))):
                    k = 2 * i + 1
                    y = y + (4.0 / (np.pi * k)) * np.sin(k * x)
                return y

            terms = ValueTracker(1)

            curve = always_redraw(lambda: axes.plot(
                lambda x: partial(x, round(terms.get_value())),
                color=MANIM_BLUE, stroke_width=4))

            self.play(Create(axes), Write(labels), Write(title), run_time=1.5)
            self.play(Create(curve), run_time=1.0)
            for target in (2, 3, 4, 6, 8, 12):
                self.play(terms.animate.set_value(target), run_time=1.1)
            self.wait(1.0)


def render_manim() -> None:
    """Render the optional Manim animation into figures/."""
    if not MANIM_AVAILABLE:
        print("[skip] Manim is not installed. Install it with: pip install manim")
        return

    media_dir = Path("media")
    cmd = [
        sys.executable, "-m", "manim", "-qm", "--disable_caching",
        "--media_dir", str(media_dir),
        Path(__file__).name, "FourierSynthesis",
    ]
    print("[run]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[warn] Manim rendering failed: {exc}")
        return

    hits = sorted(media_dir.rglob("FourierSynthesis.mp4"))
    if not hits:
        print("[warn] Could not locate the rendered animation.")
        return

    target = FIG_DIR / "fourier_synthesis.mp4"
    shutil.copy2(hits[-1], target)
    print(f"[ok] {target}")


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    print("Building static SVG figures ...")
    fig_signal_types()
    fig_sampling_quantization()
    fig_aliasing()
    fig_acquisition_chain()
    fig_wheatstone_bridge()
    fig_wheatstone_response()
    fig_noise_spectrum()
    fig_snr()
    fig_dsp_cycle()
    fig_dsp_filtering()
    fig_dft_time_freq()
    fig_dft_vs_fft()
    fig_inverse_ft()
    fig_data_presentation()

    print(f"\nAll static figures written to: {FIG_DIR.resolve()}")

    if "--manim" in sys.argv:
        render_manim()
    else:
        print("Manim animation skipped. Run 'python lecture.py --manim' to build it.")


if __name__ == "__main__":
    main()