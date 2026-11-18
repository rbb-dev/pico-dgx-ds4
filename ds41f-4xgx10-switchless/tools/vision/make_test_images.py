#!/usr/bin/env python3
"""The vision exactness suite's images, generated from a fixed seed (nothing to fetch, the same bytes every time).

    python3 make_test_images.py OUT_DIR [--seed N]

Writes into OUT_DIR:
  noise_544, noise_768        uniform noise (every pixel a free value: the worst case for rounding)
  gradient_768                a diagonal gradient with a different slope per channel
  checker_770x1022            an asymmetric checkerboard (cells of 37 x 23 pixels, one marked corner): one axis is not
                              a multiple of 3 patches
  square_1302                 smooth concentric and sinusoidal structure on 1,302 x 1,302 pixels
  strip_42x42882              a strip that plans to 3 x 3,063 patches; tall_42x42882 the same turned
  card_01 .. card_10          known-answer cards (digits in a TrueType font, counted shapes, a bar chart, a table, a
                              flag, positions, arithmetic, a word, a colour) 768 x 512
  cases.json                  the cards' questions and answers, in the shape tools/vision/cases.json has (imgtest.py reads it)
  MANIFEST.json               seed, the font used and each image's size and SHA-256

The sizes plan, through the reference's own image_processor.plan_image_grid, to 39 x 39, 55 x 55, 73 x 55, 3 x 3,063 and
(the tall one, capped by the 1,024-token limit to a single aligner column) 1,533 x 3 patches: 1,521 / 3,025 / 4,015 /
9,189 / 4,599 patches.
"""
import argparse
import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_SEED = 20261121
CARD = (768, 512)
FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 160, 60), "yellow": (240, 210, 30),
          "purple": (130, 40, 160), "white": (245, 245, 245), "black": (20, 20, 20), "orange": (240, 130, 20)}


def font(size):
    """A TrueType font at `size`: DejaVu Sans when the machine has it, else the one Pillow bundles."""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size), os.path.basename(path)
    return ImageFont.load_default(size=size), "pillow-default"


def font_name():
    return font(20)[1]


# ---------------------------------------------------------------------------------------------------------------
# structured and random images
# ---------------------------------------------------------------------------------------------------------------


def noise(w, h, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8), "RGB")


def gradient(w, h):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    d = (x / max(w - 1, 1) + y / max(h - 1, 1)) / 2
    a = np.stack([d, (d * 3) % 1.0, 1 - d], axis=-1)
    return Image.fromarray((a * 255).astype(np.uint8), "RGB")


def checker(w, h, cw=37, ch=23):
    y, x = np.mgrid[0:h, 0:w]
    c = ((x // cw + y // ch) % 2).astype(np.uint8)
    a = np.where(c[..., None] == 1, np.array([235, 235, 235], np.uint8), np.array([25, 25, 25], np.uint8)).astype(np.uint8)
    a[: 2 * ch, : 3 * cw] = (200, 40, 40)          # the marked corner: top-left is not like the others
    a[-ch:, -2 * cw:] = (40, 40, 200)
    return Image.fromarray(a, "RGB")


def smooth(w, h):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.hypot(x - w / 3, y - h / 2)
    a = np.stack([0.5 + 0.5 * np.sin(r / 23.0), 0.5 + 0.5 * np.sin(x / 61.0 + y / 97.0), 0.5 + 0.5 * np.cos(r / 41.0)], axis=-1)
    return Image.fromarray((a * 255).astype(np.uint8), "RGB")


def strip(w, h):
    """Long and thin: a gradient along the long axis under bars whose width grows (positions are distinguishable)."""
    long_axis = max(w, h)
    t = np.linspace(0.0, 1.0, long_axis)
    bars = ((np.arange(long_axis) // 211) % 2).astype(np.float64)
    c = np.stack([t, bars, 1 - t], axis=-1)  # [long, 3]
    a = np.repeat(c[None, :, :], min(w, h), axis=0) if w >= h else np.repeat(c[:, None, :], min(w, h), axis=1)
    return Image.fromarray((a * 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------------------------------------------
# the cards
# ---------------------------------------------------------------------------------------------------------------


def blank():
    return Image.new("RGB", CARD, (250, 250, 250))


def text_center(d, xy, s, f, fill=(10, 10, 10)):
    d.text(xy, s, font=f, fill=fill, anchor="mm")


def card_digits(rng):
    n = "".join(str(int(v)) for v in rng.integers(0, 10, 4))
    n = str(1 + int(n[0]) % 9) + n[1:]
    im = blank()
    d = ImageDraw.Draw(im)
    text_center(d, (CARD[0] // 2, CARD[1] // 2), n, font(220)[0])
    return im, "What number is written in this image? Answer with just the number.", [n]


def card_count_red(rng):
    k = int(rng.integers(4, 10))
    im = blank()
    d = ImageDraw.Draw(im)
    cells = [(i, j) for i in range(5) for j in range(3)]
    rng.shuffle(cells)
    for idx, (i, j) in enumerate(cells[:k + 4]):
        cx, cy = 90 + i * 145, 90 + j * 165
        d.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], fill=COLORS["red"] if idx < k else COLORS["blue"])
    return im, "How many red circles are in this image? Answer with just the number.", [str(k)]


def card_shapes(rng):
    sq, tri = int(rng.integers(2, 7)), int(rng.integers(2, 7))
    im = blank()
    d = ImageDraw.Draw(im)
    cells = [(i, j) for i in range(5) for j in range(3)]
    rng.shuffle(cells)
    for idx, (i, j) in enumerate(cells[:sq + tri]):
        cx, cy = 90 + i * 145, 90 + j * 165
        if idx < sq:
            d.rectangle([cx - 40, cy - 40, cx + 40, cy + 40], fill=COLORS["blue"])
        else:
            d.polygon([(cx, cy - 46), (cx - 46, cy + 40), (cx + 46, cy + 40)], fill=COLORS["green"])
    return im, "How many blue squares and how many green triangles are there? Answer as 'squares: N, triangles: M'.", \
        [str(sq), str(tri)]


def card_bar_chart(rng):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    vals = [int(v) * 5 for v in rng.choice(np.arange(3, 14), 5, replace=False)]
    im = blank()
    d = ImageDraw.Draw(im)
    f, small = font(30)[0], font(26)[0]
    top = max(vals)
    for i, (dname, v) in enumerate(zip(days, vals)):
        x0 = 70 + i * 135
        h = int(330 * v / top)
        d.rectangle([x0, 430 - h, x0 + 90, 430], fill=COLORS["blue"])
        text_center(d, (x0 + 45, 430 - h - 22), str(v), small)
        text_center(d, (x0 + 45, 462), dname, f)
    best = days[int(np.argmax(vals))]
    return im, "In this bar chart, which day has the most orders and roughly how many? Answer as 'day, number'.", \
        [best, str(top)]


def card_table(rng):
    cities = ["Osaka", "Lyon", "Perth", "Quito", "Turin", "Hanoi"]
    idx = rng.choice(len(cities), 4, replace=False)
    pops = rng.choice(np.arange(11, 99), 4, replace=False)
    im = blank()
    d = ImageDraw.Draw(im)
    f = font(40)[0]
    d.rectangle([60, 40, 708, 100], fill=(60, 60, 60))
    d.text((90, 70), "City", font=f, fill=(250, 250, 250), anchor="lm")
    d.text((420, 70), "Population (k)", font=f, fill=(250, 250, 250), anchor="lm")
    for r, (ci, p) in enumerate(zip(idx, pops)):
        y = 100 + r * 85
        d.rectangle([60, y, 708, y + 85], outline=(60, 60, 60), width=3)
        d.text((90, y + 42), cities[ci], font=f, fill=(10, 10, 10), anchor="lm")
        d.text((420, y + 42), str(int(p) * 10), font=f, fill=(10, 10, 10), anchor="lm")
    return im, "According to this table, which city has the largest population? Answer with the city name.", \
        [cities[idx[int(np.argmax(pops))]]]


def card_flag(rng):
    names = list(rng.choice(["red", "white", "green", "blue", "yellow", "black"], 3, replace=False))
    im = blank()
    d = ImageDraw.Draw(im)
    w = CARD[0] // 3
    for i, nme in enumerate(names):
        d.rectangle([i * w, 0, (i + 1) * w, CARD[1]], fill=COLORS[str(nme)])
    return im, "Name the three colours of the vertical stripes from left to right.", [str(n) for n in names]


def card_position(rng):
    corners = {"top-left": (110, 110), "top-right": (658, 110), "bottom-left": (110, 402), "bottom-right": (658, 402)}
    names = list(corners)
    ci, si = rng.choice(len(names), 2, replace=False)
    im = blank()
    d = ImageDraw.Draw(im)
    cx, cy = corners[names[ci]]
    d.ellipse([cx - 60, cy - 60, cx + 60, cy + 60], fill=COLORS["yellow"])
    sx, sy = corners[names[si]]
    d.rectangle([sx - 60, sy - 60, sx + 60, sy + 60], fill=COLORS["purple"])
    ans = names[ci]
    return im, ("There is a yellow circle and a purple square. In which corner of the image is the yellow circle "
                "(top-left, top-right, bottom-left, bottom-right)?"), [[ans, ans.replace("-", " ")]]


def card_math(rng):
    a, b = int(rng.integers(21, 90)), int(rng.integers(21, 90))
    im = blank()
    d = ImageDraw.Draw(im)
    text_center(d, (CARD[0] // 2, CARD[1] // 2), "%d + %d" % (a, b), font(150)[0])
    return im, "Solve the arithmetic problem written in the image. Answer with just the number.", [str(a + b)]


def card_word(rng):
    words = ["LANTERN", "HARBOUR", "MONSOON", "PEBBLE", "CANYON", "FALCON", "THIMBLE", "GLACIER"]
    w = str(rng.choice(words))
    im = blank()
    d = ImageDraw.Draw(im)
    text_center(d, (CARD[0] // 2, CARD[1] // 2), w, font(120)[0])
    return im, "What word is written in this image? Answer with just the word.", [w]


def card_colour(rng):
    names = ["red", "blue", "green", "purple", "orange"]
    pick = str(rng.choice(names))
    im = blank()
    d = ImageDraw.Draw(im)
    d.polygon([(384, 60), (150, 450), (618, 450)], fill=COLORS[pick])
    d.ellipse([620, 40, 720, 140], fill=COLORS["black"])
    return im, "What colour is the large triangle? Answer with just the colour.", [pick]


CARDS = [("card_01_digits", card_digits), ("card_02_count_red", card_count_red), ("card_03_shapes", card_shapes),
         ("card_04_bar_chart", card_bar_chart), ("card_05_table", card_table), ("card_06_flag", card_flag),
         ("card_07_position", card_position), ("card_08_math", card_math), ("card_09_word", card_word),
         ("card_10_colour", card_colour)]


# ---------------------------------------------------------------------------------------------------------------
# all of it
# ---------------------------------------------------------------------------------------------------------------


def make_all(out, seed=DEFAULT_SEED):
    """Writes every image, cases.json and MANIFEST.json into `out`; returns the image stems in order."""
    os.makedirs(out, exist_ok=True)
    images = [
        ("noise_544", noise(544, 544, seed + 1)), ("noise_768", noise(768, 768, seed + 2)),
        ("gradient_768", gradient(768, 768)), ("checker_770x1022", checker(770, 1022)),
        ("square_1302", smooth(1302, 1302)), ("strip_42x42882", strip(42882, 42)), ("tall_42x42882", strip(42, 42882)),
    ]
    cases = []
    for k, (stem, fn) in enumerate(CARDS):
        im, q, a = fn(np.random.default_rng(seed * 100 + k))
        images.append((stem, im))
        cases.append([stem + ".png", q, a])
    manifest = {"seed": seed, "font": font_name(), "images": {}}
    for stem, im in images:
        path = os.path.join(out, stem + ".png")
        im.save(path, format="PNG", compress_level=6)
        manifest["images"][stem] = {"size": list(im.size), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(cases, open(os.path.join(out, "cases.json"), "w"), indent=1)
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return [s for s, _ in images]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = p.parse_args()
    names = make_all(a.out, a.seed)
    print("wrote %d images, cases.json and MANIFEST.json to %s (font %s)" % (len(names), a.out, font_name()))


if __name__ == "__main__":
    main()
