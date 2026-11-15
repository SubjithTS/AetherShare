"""Visual secret sharing (Naor-Shamir) for binary images.

Every secret pixel is expanded into m subpixels in each of the n shares.
Stacking shares means OR-ing them: a subpixel is black if it is black in any
stacked share. Black = 1, white = 0 everywhere in this file.

Two kinds of scheme are supported:

* (k, n) with k == n : basis matrices built from column parity.
    S0 = all n-bit columns with even weight, S1 = all with odd weight, m = 2^(n-1).
    Any n-1 rows of S0 and S1 contain exactly the same set of columns, so
    n-1 shares carry no information. All n shares stacked: a white pixel keeps
    one white subpixel, a black pixel is fully black.

* (2, n): S0 has a single black column shared by all rows, S1 is the identity
    matrix, m = n. One share is a single black subpixel at a random place
    either way. Any two shares stacked: white = 1 black subpixel,
    black = 2 black subpixels.

For each secret pixel the same random column permutation is applied to the
whole matrix, and row i becomes the pixel in share i.
"""
import itertools
import os
import numpy as np

SUPPORTED = [(2, 2), (2, 3), (2, 4), (3, 3)]


def basis_matrices(k, n):
    """Return S0, S1 with shape (n, m)."""
    if (k, n) not in SUPPORTED:
        raise ValueError(f"Unsupported scheme {k}-of-{n}")
    if k == n:
        columns = list(itertools.product([0, 1], repeat=n))
        even = [c for c in columns if sum(c) % 2 == 0]
        odd = [c for c in columns if sum(c) % 2 == 1]
        return np.array(even, dtype=np.uint8).T, np.array(odd, dtype=np.uint8).T
    s0 = np.zeros((n, n), dtype=np.uint8)
    s0[:, 0] = 1
    s1 = np.eye(n, dtype=np.uint8)
    return s0, s1


def expansion(k, n):
    return basis_matrices(k, n)[0].shape[1]


def stacked_lightness(k, n):
    """Fraction of white subpixels when exactly k shares are stacked.

    Returns (white_pixel, black_pixel). Used by the viewer to stretch contrast.
    """
    s0, s1 = basis_matrices(k, n)
    rows = list(range(k))
    white = 1 - np.bitwise_or.reduce(s0[rows], axis=0).mean()
    black = 1 - np.bitwise_or.reduce(s1[rows], axis=0).mean()
    return float(white), float(black)


def _random_permutation_indices(count, m):
    """Pick `count` uniformly random permutations of range(m) using OS randomness."""
    perms = np.array(list(itertools.permutations(range(m))), dtype=np.int64)
    raw = np.frombuffer(os.urandom(count * 4), dtype=np.uint32)
    return perms[raw % len(perms)]


def split(secret, k, n):
    """Split a binary image (h, w) of 0/1 values into n shares of shape (h, w*m)."""
    secret = np.asarray(secret, dtype=np.uint8)
    h, w = secret.shape
    s0, s1 = basis_matrices(k, n)
    m = s0.shape[1]
    base = np.stack([s0, s1])                       # (2, n, m)
    pixel_matrix = base[secret]                     # (h, w, n, m)
    perms = _random_permutation_indices(h * w, m).reshape(h, w, m)
    index = np.broadcast_to(perms[:, :, None, :], (h, w, n, m))
    shuffled = np.take_along_axis(pixel_matrix, index, axis=3)
    # (h, w, n, m) -> (n, h, w*m)
    return [shuffled[:, :, i, :].reshape(h, w * m) for i in range(n)]


def stack(shares):
    """Overlay shares the way transparencies would: black wins."""
    return np.bitwise_or.reduce(np.stack(shares), axis=0)


def to_gray(stacked, m):
    """Average each group of m subpixels back to one gray pixel (0 black .. 1 white)."""
    h, wm = stacked.shape
    return 1 - stacked.reshape(h, wm // m, m).mean(axis=2)
