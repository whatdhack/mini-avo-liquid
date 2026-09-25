"""Render assets/demo.gif: an animated overview of the MiniAVO-Liquid loop.

Illustrative only (no real results): candidates are drafted, filtered by the Python syntax
check and the GPU-less Triton compile, scored statically, and the best becomes the next parent.

    python assets/make_demo_gif.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
BG = (250, 250, 247)
INK = (38, 38, 36)
MUTED = (120, 120, 114)
LINE = (205, 205, 198)
ACCENT = (37, 99, 235)
OK = (22, 142, 86)
BAD = (200, 60, 50)
GOLD = (196, 142, 20)
CARD = (255, 255, 255)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE, F_H, F_B, F_S = font(26, True), font(17, True), font(15), font(13)

# Pipeline columns: (x center, title, subtitle lines)
STAGES = [
    (95, "Parent", ["seed: naive or", "web (Nimble)"]),
    (265, "Liquid LFM", ["drafts candidate", "kernels (OpenRouter)"]),
    (440, "1. Syntax", ["ast.parse"]),
    (610, "2. Triton compile", ["sm_90 / sm_100", "no GPU needed"]),
    (790, "3. Static score", ["regs · spills · occupancy", "tensor cores · TMA · WS"]),
]
LANES = [205, 265, 325, 385]  # y of each candidate lane
# Where each candidate stops: 2 = fails syntax, 3 = fails compile, 4 = scored
FATE = [4, 2, 4, 3]
BARS = [0.62, None, 0.84, None]  # illustrative relative score bars (no real numbers)
BEST = 2


def base(gen_label, caption, stages=True):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((32, 22), "MiniAVO-Liquid", font=F_TITLE, fill=INK)
    title_right = d.textbbox((32, 22), "MiniAVO-Liquid", font=F_TITLE)[2]
    d.text((title_right + 16, 30), "GPU-less evolutionary search for Triton kernels", font=F_B, fill=MUTED)
    d.text((W - 32, 30), gen_label, font=F_H, fill=ACCENT, anchor="ra")
    for x, title, sub in (STAGES if stages else []):
        d.text((x, 92), title, font=F_H, fill=INK, anchor="ma")
        for i, s in enumerate(sub):
            d.text((x, 116 + 18 * i), s, font=F_S, fill=MUTED, anchor="ma")
        d.line([(x, 170), (x, 420)], fill=LINE, width=1)
    d.rounded_rectangle([32, 452, W - 32, 508], radius=10, fill=CARD, outline=LINE)
    d.text((W // 2, 480), caption, font=F_B, fill=INK, anchor="mm")
    return img, d


def parent_card(d, label="parent kernel", color=INK):
    d.rounded_rectangle([40, 265, 150, 325], radius=8, fill=CARD, outline=color, width=2)
    d.text((95, 295), label, font=F_S, fill=color, anchor="mm")


def candidate(d, x, y, idx, state):
    color = {"new": ACCENT, "ok": OK, "bad": BAD, "best": GOLD}[state]
    d.rounded_rectangle([x - 42, y - 20, x + 42, y + 20], radius=7, fill=CARD, outline=color, width=2)
    mark = {"new": "", "ok": " ✓", "bad": " ✗", "best": " ★"}[state]
    d.text((x, y), f"draft {idx + 1}{mark}", font=F_S, fill=color, anchor="mm")


def score_bar(d, y, frac, best):
    x0 = 842
    d.rounded_rectangle([x0, y - 7, x0 + 90, y + 7], radius=4, fill=(236, 236, 230))
    d.rounded_rectangle([x0, y - 7, x0 + int(90 * frac), y + 7], radius=4, fill=GOLD if best else OK)


def frames():
    out = []
    # 1. parent + fan-out from the Liquid model
    for gen in (1, 2):
        g = f"generation {gen}"
        img, d = base(g, "Start from the current parent kernel")
        parent_card(d)
        out.append((img, 900))
        img, d = base(g, "Liquid drafts N candidates in parallel (generation fan-out, planned; CLI today: N = 1)")
        parent_card(d)
        for i, y in enumerate(LANES):
            d.line([(150, 295), (223, y)], fill=MUTED, width=2)
            candidate(d, STAGES[1][0], y, i, "new")
        out.append((img, 1500))
        # 2. move through the checks stage by stage
        captions = {2: "Python syntax check drops unparseable code",
                    3: "Ahead-of-time Triton compile drops made-up tl.* APIs and type errors",
                    4: "Survivors get a static score from the compiled PTX / cubin"}
        for stage in (2, 3, 4):
            img, d = base(g, captions[stage])
            parent_card(d)
            for i, y in enumerate(LANES):
                # A candidate stops at the stage it fails; everything else advances
                column = min(stage, FATE[i])
                state = "bad" if FATE[i] <= stage and FATE[i] < 4 else "ok"
                candidate(d, STAGES[column][0] if column < 4 else STAGES[4][0] - 30, y, i, state)
                if stage == 4 and FATE[i] == 4:
                    score_bar(d, y, BARS[i], False)
            out.append((img, 1300))
        # 3. pick the elite and loop back
        img, d = base(g, "Highest static score (strictly better than the parent) becomes the new parent")
        parent_card(d, "old parent", MUTED)
        for i, y in enumerate(LANES):
            if FATE[i] == 4:
                candidate(d, STAGES[4][0] - 30, y, i, "best" if i == BEST else "ok")
                score_bar(d, y, BARS[i], i == BEST)
            else:
                candidate(d, STAGES[FATE[i]][0], y, i, "bad")
        y = LANES[BEST]
        d.line([(STAGES[4][0] - 30, y + 20), (STAGES[4][0] - 30, 432), (95, 432), (95, 325)], fill=GOLD, width=3)
        d.polygon([(95, 325), (88, 337), (102, 337)], fill=GOLD)
        d.text((440, 424), "next generation", font=F_S, fill=GOLD, anchor="mb")
        out.append((img, 1800))
    # 4. closing card
    img, d = base("", "No real GPU: compiled for the target GPU, then scored statically", stages=False)
    lines = [("Checks", "Python syntax → Triton compile (sm_90 / sm_100) → launch limits"),
             ("Static score", "registers, spills, occupancy, load width, tensor cores (wgmma / tcgen05),"),
             ("", "TMA, warp specialization, tile reuse"),
             ("Output", "lineage JSON + submission script for the best kernel")]
    for i, (k, v) in enumerate(lines):
        d.text((90, 150 + 50 * i), k, font=F_H, fill=ACCENT)
        d.text((250, 152 + 50 * i), v, font=F_B, fill=INK)
    out.append((img, 3000))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fr = frames()
    imgs = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f, _ in fr]
    path = os.path.join(here, "demo.gif")
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=[ms for _, ms in fr], loop=0, optimize=True)
    print(f"wrote {path} ({len(imgs)} frames, {os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
