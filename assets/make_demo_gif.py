"""Render assets/demo.gif: an animated overview of the MiniAVO-Liquid loop.

Illustrative only (no real results): candidates are drafted, filtered by the Python syntax
check and the GPU-less Triton compile, scored statically, and every Nth generation the
survivors are run on a real GPU — checked against a torch reference, then timed. The best
becomes the next parent. What the checks and the GPU run found is turned into instructions
(feedback memory: this run, a local file, or RawTree) for the next generation's prompt.

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


F_TITLE, F_H, F_B, F_S = font(26, True), font(16, True), font(15), font(12)

# Pipeline columns: (x center, title, subtitle lines)
STAGES = [
    (95, "Parent", ["seed: naive or", "web (Nimble)"]),
    (228, "LLM agent", ["drafts candidates", "OpenRouter / Vultr"]),
    (348, "1. Syntax", ["ast.parse"]),
    (474, "2. Compile", ["Triton, no GPU", "sm_80 … sm_121"]),
    (625, "3. Static score", ["regs · occupancy", "tensor cores · tiles"]),
    (830, "4. GPU run", ["every Nth generation", "correctness, then time"]),
]
SYNTAX, COMPILE, SCORED, TIMED = 2, 3, 4, 5
LANES = [205, 265, 325, 385]
# Where each candidate stops. Draft 3 compiles and scores highest, and is still wrong on the GPU.
FATE = [TIMED, SYNTAX, TIMED, COMPILE]
BARS = [0.88, None, 1.0, None]          # illustrative static scores (no real numbers)
GPU = ["1602 µs", None, "wrong", None]  # illustrative GPU outcome
BEST = 0


def base(gen_label, caption, stages=True):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((32, 22), "MiniAVO-Liquid", font=F_TITLE, fill=INK)
    title_right = d.textbbox((32, 22), "MiniAVO-Liquid", font=F_TITLE)[2]
    d.text((title_right + 16, 31), "evolutionary search for Triton kernels: compile-checked, GPU-timed",
           font=F_B, fill=MUTED)
    d.text((W - 32, 31), gen_label, font=F_H, fill=ACCENT, anchor="ra")
    for x, title, sub in (STAGES if stages else []):
        d.text((x, 92), title, font=F_H, fill=INK, anchor="ma")
        for i, s in enumerate(sub):
            d.text((x, 115 + 17 * i), s, font=F_S, fill=MUTED, anchor="ma")
        d.line([(x, 170), (x, 420)], fill=LINE, width=1)
    d.rounded_rectangle([32, 452, W - 32, 508], radius=10, fill=CARD, outline=LINE)
    d.text((W // 2, 480), caption, font=F_B, fill=INK, anchor="mm")
    return img, d


def parent_card(d, label="parent kernel", color=INK):
    d.rounded_rectangle([40, 265, 150, 325], radius=8, fill=CARD, outline=color, width=2)
    d.text((95, 295), label, font=F_S, fill=color, anchor="mm")


def candidate(d, x, y, idx, state):
    color = {"new": ACCENT, "ok": OK, "bad": BAD, "best": GOLD}[state]
    d.rounded_rectangle([x - 40, y - 19, x + 40, y + 19], radius=7, fill=CARD, outline=color, width=2)
    mark = {"new": "", "ok": " ✓", "bad": " ✗", "best": " ★"}[state]
    d.text((x, y), f"draft {idx + 1}{mark}", font=F_S, fill=color, anchor="mm")


def score_bar(d, y, frac, best):
    x0 = 682
    d.rounded_rectangle([x0, y - 6, x0 + 72, y + 6], radius=4, fill=(236, 236, 230))
    d.rounded_rectangle([x0, y - 6, x0 + int(72 * frac), y + 6], radius=4, fill=GOLD if best else OK)


def gpu_label(d, y, text, color):
    d.text((876, y), text, font=F_S, fill=color, anchor="lm")


def arrow(d, start, end, color, width=2):
    d.line([start, end], fill=color, width=width)
    (x0, y0), (x1, y1) = start, end
    length = max(1.0, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    d.polygon([(x1, y1), (x1 - 11 * ux + 5 * uy, y1 - 11 * uy - 5 * ux), (x1 - 11 * ux - 5 * uy, y1 - 11 * uy + 5 * ux)],
              fill=color)


def feedback_frame(gen_label, benched):
    """Findings -> feedback memory -> instructions in the next prompt (illustrative rule types)."""
    img, d = base(gen_label, "What the checks%s found becomes instructions for the next prompt"
                  % (" and the GPU run" if benched else ""), stages=False)
    sources = [("draft 2 ✗  syntax error", BAD), ("draft 4 ✗  compile error", BAD)]
    sources += ([("draft 3 ✗  wrong on the GPU", BAD), ("draft 1 ✓  timed, kept", OK)] if benched
                else [("draft 1 ✓  near miss", OK), ("parent: static weak spots", INK)])
    for i, (label, color) in enumerate(sources):
        y = 150 + 62 * i
        d.rounded_rectangle([40, y - 20, 268, y + 20], radius=7, fill=CARD, outline=color, width=2)
        d.text((154, y), label, font=F_S, fill=color, anchor="mm")
        arrow(d, (270, y), (336, 205 + 26 * i), MUTED)
    d.rounded_rectangle([340, 188, 566, 298], radius=10, fill=CARD, outline=ACCENT, width=2)
    d.text((453, 210), "Feedback memory", font=F_H, fill=ACCENT, anchor="mm")
    d.text((453, 237), "local: this run", font=F_S, fill=INK, anchor="mm")
    d.text((453, 258), "file: .miniavo_memory.jsonl", font=F_S, fill=INK, anchor="mm")
    d.text((453, 279), "RawTree: across runs", font=F_S, fill=INK, anchor="mm")
    arrow(d, (568, 243), (604, 243), ACCENT, 3)
    d.rounded_rectangle([606, 112, 930, 388], radius=10, fill=CARD, outline=LINE)
    d.text((622, 128), "Next prompt: rules and timings" if benched else "Next prompt: rules",
           font=F_H, fill=INK)
    rules = ["tl.ceil_div doesn't exist → tl.cdiv", "Write Python, not C (no 0.0f)",
             "No manual shared memory in Triton", "Shared memory over the per-SM limit"]
    rules += (["Zero the buffer you accumulate into", "The kernel to beat ran in 1602 µs",
               "These strategies were already timed"] if benched
              else ["Loads are narrow → larger BLOCK", "Scored no higher than the parent"])
    for i, rule in enumerate(rules):
        d.text((622, 164 + 31 * i), "• " + rule, font=F_S, fill=INK)
    d.text((453, 325), "repeats merged: \"seen N×\"", font=F_S, fill=MUTED, anchor="mm")
    return img, d


def frames():
    out = []
    for gen in (1, 2):
        g = f"generation {gen}"
        benched = gen == 2  # --gpu-bench-every 2: only even generations reach the GPU
        img, d = base(g, "Start from the current parent kernel")
        parent_card(d)
        out.append((img, 900))
        caption = ("The agent drafts N candidates in parallel (fan-out planned; CLI today: N = 1)"
                   if gen == 1 else "The prompt now carries the rules and the measured timings from before")
        img, d = base(g, caption)
        parent_card(d)
        for i, y in enumerate(LANES):
            d.line([(150, 295), (191, y)], fill=MUTED, width=2)
            candidate(d, STAGES[1][0], y, i, "new")
        if gen > 1:
            d.rounded_rectangle([173, 414, 293, 438], radius=12, fill=ACCENT)
            d.text((233, 426), "+ feedback", font=F_S, fill=CARD, anchor="mm")
        out.append((img, 1500))

        captions = {SYNTAX: "Python syntax check drops unparseable code",
                    COMPILE: "Ahead-of-time Triton compile drops made-up tl.* APIs and type errors",
                    SCORED: "Survivors get a static score from the compiled PTX / cubin"}
        for stage in (SYNTAX, COMPILE, SCORED):
            img, d = base(g, captions[stage])
            parent_card(d)
            for i, y in enumerate(LANES):
                column = min(stage, FATE[i])
                state = "bad" if FATE[i] <= stage and FATE[i] < SCORED else "ok"
                candidate(d, STAGES[column][0], y, i, state)
                if stage == SCORED and FATE[i] >= SCORED:
                    score_bar(d, y, BARS[i], False)
            out.append((img, 1300))

        if benched:
            img, d = base(g, "Every Nth generation the survivors run on a real GPU: correctness first, then time")
            parent_card(d)
            for i, y in enumerate(LANES):
                if FATE[i] >= SCORED:
                    wrong = GPU[i] == "wrong"
                    candidate(d, STAGES[5][0], y, i, "bad" if wrong else "ok")
                    gpu_label(d, y, "✗ wrong" if wrong else GPU[i], BAD if wrong else OK)
                else:
                    candidate(d, STAGES[FATE[i]][0], y, i, "bad")
            out.append((img, 2200))

        img, d = base(g, "Fastest measured time wins — draft 3 scored highest and was wrong on the GPU"
                      if benched else "Highest static score (strictly better than the parent) becomes the new parent")
        parent_card(d, "old parent", MUTED)
        for i, y in enumerate(LANES):
            if FATE[i] >= SCORED:
                at_gpu = benched
                x = STAGES[5][0] if at_gpu else STAGES[4][0]
                wrong = benched and GPU[i] == "wrong"
                candidate(d, x, y, i, "bad" if wrong else ("best" if i == BEST else "ok"))
                if not at_gpu:
                    score_bar(d, y, BARS[i], i == BEST)
                elif not wrong:
                    gpu_label(d, y, GPU[i], GOLD if i == BEST else OK)
                else:
                    gpu_label(d, y, "✗ wrong", BAD)
            else:
                candidate(d, STAGES[FATE[i]][0], y, i, "bad")
        # Drop just left of the elite's column: straight down would cut through the card in the
        # lane below, and the right margin holds the measured-time labels
        y, x = LANES[BEST], STAGES[5][0] if benched else STAGES[4][0]
        drop = x - 60
        d.line([(x - 40, y), (drop, y), (drop, 432), (95, 432), (95, 325)], fill=GOLD, width=3)
        d.polygon([(95, 325), (88, 337), (102, 337)], fill=GOLD)
        d.text((480, 424), "next generation", font=F_S, fill=GOLD, anchor="mb")
        out.append((img, 1800))
        out.append((feedback_frame(g, benched)[0], 2600))

    img, d = base("", "No GPU needed to search; one on the machine turns the proxy into a measurement",
                  stages=False)
    lines = [("Problems", "matmul_v2 · vectorsum_v2 · cholesky · trimul_alphafold3"),
             ("Checks", "Python syntax → Triton compile (sm_80 … sm_121) → launch limits"),
             ("Static score", "registers band, spills, occupancy, load width, tensor cores, tiles"),
             ("GPU run", "--gpu-bench-every N: torch reference, extra shapes, then do_bench"),
             ("Feedback", "findings and timings → next prompt; memory: run, file or RawTree"),
             ("Output", "lineage JSON + submission script for the best kernel")]
    for i, (k, v) in enumerate(lines):
        d.text((90, 142 + 46 * i), k, font=F_H, fill=ACCENT)
        d.text((260, 143 + 46 * i), v, font=F_B, fill=INK)
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
