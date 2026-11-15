"""Checks the two claims from the slides: k shares rebuild the secret, fewer reveal nothing.

Run:  python verify.py
"""
import itertools
import numpy as np
from PIL import Image, ImageDraw
import vss

def make_secret(w=400, h=200):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 180, 180], fill=0)
    d.ellipse([220, 40, 380, 160], fill=0)
    return (np.asarray(img) < 128).astype(np.uint8)


def region_stats(gray, secret):
    """Mean gray level over secret-white pixels and over secret-black pixels."""
    return gray[secret == 0].mean(), gray[secret == 1].mean()


secret = make_secret()
for k, n in vss.SUPPORTED:
    m = vss.expansion(k, n)
    shares = vss.split(secret, k, n)
    print(f"\n{k}-of-{n}  (each pixel becomes {m} subpixels)")

    for r in range(1, n + 1):
        gaps = []
        for combo in itertools.combinations(range(n), r):
            gray = vss.to_gray(vss.stack([shares[i] for i in combo]), m)
            white_mean, black_mean = region_stats(gray, secret)
            gaps.append(white_mean - black_mean)
        gap = np.mean(gaps)
        verdict = "readable" if gap > 0.05 else "no information"
        need = "" if r >= k else "  (fewer than k)"
        print(f"  {r} share(s) stacked: brightness gap between background and ink = {gap:.3f} -> {verdict}{need}")
