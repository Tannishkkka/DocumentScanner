"""Offline test of the SmartScan pipeline.

1. Builds synthetic "photos" of a known document under different conditions
   (the clean document and the true corners are known, so metrics are valid).
2. Runs detection + perspective correction + enhancement on each one.
3. Runs detection on any real photos placed directly in sample_images/.
4. Checks that invalid files are rejected.

Run:  python test_pipeline.py
Outputs are written to sample_images/synthetic/ and results/.
"""
import os
import time

import cv2
import numpy as np

from modules.preprocessing import load_image_from_bytes, prepare_for_detection
from modules.document_detection import detect_document, draw_boundary, refine_corners
from modules.perspective_transform import four_point_transform, order_points
from modules.image_enhancement import grayscale_scan, black_and_white_scan, enhanced_scan
from modules.evaluation import evaluate
from modules.export_document import to_pdf_bytes

SYNTHETIC_DIR = os.path.join("sample_images", "synthetic")
RESULTS_DIR = "results"
SCENE_W, SCENE_H = 1200, 1600
rng = np.random.default_rng(7)


def make_document():
    """A clean white A4-shaped page with a title, text lines and a table."""
    page = np.full((1100, 850, 3), 255, np.uint8)
    cv2.putText(page, "SMARTSCAN TEST DOCUMENT", (70, 110), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), 2)
    words = ["image", "video", "processing", "edge", "contour", "warp", "filter", "pixel",
             "threshold", "contrast", "scan", "matrix", "kernel", "noise"]
    y = 190
    for _ in range(18):
        line = " ".join(rng.choice(words, 6))
        cv2.putText(page, line, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
        y += 38
    top = 900
    for row in range(4):
        for col in range(3):
            x1, y1 = 70 + col * 240, top + row * 40
            cv2.rectangle(page, (x1, y1), (x1 + 240, y1 + 40), (0, 0, 0), 2)
    cv2.rectangle(page, (580, 160), (780, 300), (180, 120, 40), -1)  # a coloured block
    return page


def make_background(kind):
    if kind == "dark":
        bg = np.full((SCENE_H, SCENE_W, 3), (55, 45, 40), np.uint8)
    elif kind == "wood":
        rows = np.arange(SCENE_H)[:, None]
        stripes = 25 * np.sin(rows / 9.0) + 10 * np.sin(rows / 37.0)
        base = np.zeros((SCENE_H, SCENE_W, 3), np.float32)
        base[:] = (40, 75, 120)
        bg = np.clip(base + stripes[..., None], 0, 255).astype(np.uint8)
    elif kind == "complex":
        bg = np.full((SCENE_H, SCENE_W, 3), (90, 90, 90), np.uint8)
        for _ in range(60):
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            x, y = int(rng.integers(0, SCENE_W)), int(rng.integers(0, SCENE_H))
            w, h = int(rng.integers(40, 300)), int(rng.integers(40, 300))
            cv2.rectangle(bg, (x, y), (x + w, y + h), color, -1)
        for _ in range(40):
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            p1 = (int(rng.integers(0, SCENE_W)), int(rng.integers(0, SCENE_H)))
            p2 = (int(rng.integers(0, SCENE_W)), int(rng.integers(0, SCENE_H)))
            cv2.line(bg, p1, p2, color, 4)
    else:  # "light": background almost as bright as the paper
        bg = np.full((SCENE_H, SCENE_W, 3), (215, 215, 215), np.uint8)
    noise = rng.normal(0, 4, bg.shape)
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


def place_document(document, background, corners):
    h, w = document.shape[:2]
    source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(source, np.float32(corners))
    warped = cv2.warpPerspective(document, matrix, (SCENE_W, SCENE_H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (SCENE_W, SCENE_H))
    scene = background.copy()
    scene[mask > 0] = warped[mask > 0]
    return scene


def apply_shadow(scene):
    gradient = np.linspace(0.4, 1.0, SCENE_W)[None, :, None]
    return np.clip(scene * gradient, 0, 255).astype(np.uint8)


def apply_low_light(scene):
    dark = scene.astype(np.float32) * 0.3 + rng.normal(0, 6, scene.shape)
    return np.clip(dark, 0, 255).astype(np.uint8)


def apply_occlusion(scene, corner):
    x, y = int(corner[0]), int(corner[1])
    cv2.ellipse(scene, (x + 40, y + 40), (150, 110), 30, 0, 360, (90, 120, 170), -1)
    return scene


CASES = [
    ("aligned", "dark", [[180, 200], [1020, 210], [1030, 1400], [170, 1390]], None),
    ("tilted", "wood", [[330, 140], [1080, 380], [880, 1460], [110, 1220]], None),
    ("perspective", "dark", [[300, 300], [900, 290], [1110, 1450], [90, 1460]], None),
    ("shadow", "dark", [[200, 180], [1010, 230], [990, 1420], [160, 1380]], apply_shadow),
    ("low_light", "dark", [[210, 220], [1000, 200], [1020, 1390], [190, 1410]], apply_low_light),
    ("complex_background", "complex", [[260, 230], [960, 260], [990, 1330], [220, 1310]], None),
    ("low_contrast_background", "light", [[200, 200], [1000, 220], [1010, 1400], [180, 1380]], None),
    ("occluded_corner", "dark", [[200, 200], [1000, 220], [1010, 1400], [180, 1380]], "occlude"),
]


def corner_error(found, truth):
    found, truth = order_points(found), order_points(truth)
    return float(np.mean(np.linalg.norm(found - truth, axis=1)))


def run_pipeline(scene):
    times = {}
    start = time.perf_counter()
    prep = prepare_for_detection(scene)
    times["preprocess"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    corners, _, method = detect_document(prep)
    times["detect"] = (time.perf_counter() - start) * 1000
    if corners is None:
        return None, method, times, None

    start = time.perf_counter()
    corners = refine_corners(scene, corners, prep["ratio"])
    times["refine"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    warped, _ = four_point_transform(scene, corners)
    times["warp"] = (time.perf_counter() - start) * 1000

    outputs = {"Colour (warped)": warped}
    for name, func in [("Grayscale", grayscale_scan), ("Black and White", black_and_white_scan),
                       ("Enhanced", enhanced_scan)]:
        start = time.perf_counter()
        outputs[name] = func(warped)
        times[name] = (time.perf_counter() - start) * 1000
    return corners, method, times, outputs


def test_synthetic():
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    document = make_document()
    cv2.imwrite(os.path.join(SYNTHETIC_DIR, "ground_truth.png"), document)

    detection_rows, quality_rows = [], []
    for name, bg_kind, true_corners, effect in CASES:
        scene = place_document(document, make_background(bg_kind), true_corners)
        if effect == "occlude":
            scene = apply_occlusion(scene, true_corners[2])
        elif effect is not None:
            scene = effect(scene)
        scene = cv2.GaussianBlur(scene, (3, 3), 0)  # mild camera blur
        cv2.imwrite(os.path.join(SYNTHETIC_DIR, f"{name}.jpg"), scene)

        # Run 3 times and report the median total time (single runs are noisy)
        totals = []
        for _ in range(3):
            corners, method, times, outputs = run_pipeline(scene)
            totals.append(sum(times.values()))
        total = float(np.median(totals))
        if corners is None:
            detection_rows.append((name, "NO", method, "-", f"{total:.1f}"))
            continue
        error = corner_error(corners, np.float32(true_corners))
        detection_rows.append((name, "yes", method, f"{error:.1f}", f"{total:.1f}"))

        cv2.imwrite(os.path.join(RESULTS_DIR, f"{name}_1_boundary.jpg"), draw_boundary(scene, corners))
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{name}_2_warped.jpg"), outputs["Colour (warped)"])
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{name}_3_bw.png"), outputs["Black and White"])
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{name}_4_enhanced.jpg"), outputs["Enhanced"])

        # Baseline: warp with the TRUE corners, to separate detection error from
        # the effects of lighting / blur / resampling
        ideal_warp, _ = four_point_transform(scene, np.float32(true_corners))
        outputs["Grayscale (true corners)"] = grayscale_scan(ideal_warp)

        for output_name, output in outputs.items():
            if output_name == "Colour (warped)":
                continue  # identical to Grayscale after metric conversion
            scores = evaluate(output, document)
            quality_rows.append((name, output_name, f"{scores['MSE']:.1f}",
                                 f"{scores['PSNR (dB)']:.2f}", f"{scores['SSIM']:.4f}"))

    print_table("DOCUMENT DETECTION (synthetic, true corners known)",
                ["Case", "Detected", "Method", "Mean corner error (px)", "Median total time (ms)"],
                detection_rows)
    print_table("QUALITY vs CLEAN GROUND-TRUTH DOCUMENT",
                ["Case", "Output", "MSE", "PSNR (dB)", "SSIM"], quality_rows)


def test_real_photos():
    files = [f for f in os.listdir("sample_images")
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print("\nNo real photos found directly in sample_images/ (skipped).")
        return
    rows = []
    for filename in sorted(files):
        with open(os.path.join("sample_images", filename), "rb") as f:
            image = load_image_from_bytes(f.read())
        if image is None:
            rows.append((filename, "unreadable", "-", "-"))
            continue
        corners, method, times, _ = run_pipeline(image)
        rows.append((filename, "yes" if corners is not None else "NO", method,
                     f"{sum(times.values()):.1f}"))
        if corners is not None:
            cv2.imwrite(os.path.join(RESULTS_DIR, f"real_{filename}_boundary.jpg"),
                        draw_boundary(image, corners))
    print_table("REAL PHOTOS (detection only, no reference available)",
                ["File", "Detected", "Method", "Total time (ms)"], rows)


def test_invalid_inputs():
    valid_jpeg = cv2.imencode(".jpg", make_background("dark"))[1].tobytes()
    cases = [("empty file", b""), ("text file renamed to .jpg", b"hello this is not an image"),
             ("truncated JPEG (first 200 bytes)", valid_jpeg[:200])]
    rows = []
    for label, data in cases:
        image = load_image_from_bytes(data)
        rows.append((label, "rejected (None)" if image is None else f"decoded {image.shape}"))
    print_table("INVALID INPUT HANDLING", ["Input", "Result"], rows)

    tiny = np.zeros((5, 5, 3), np.uint8)
    try:
        four_point_transform(tiny, np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]))
        print("Tiny region warp: NOT rejected")
    except ValueError as error:
        print(f"Tiny region warp: rejected -> {error}")

    pdf = to_pdf_bytes(make_document())
    print(f"PDF export: {len(pdf)} bytes, starts with {pdf[:5]!r}")


def test_timing(runs=5):
    """Per-stage timing on each synthetic photo: median and minimum of several runs."""
    stage_times = {}
    for name, *_ in CASES[:-1]:
        scene = cv2.imread(os.path.join(SYNTHETIC_DIR, f"{name}.jpg"))
        for _ in range(runs):
            _, _, times, _ = run_pipeline(scene)
            for stage, ms in times.items():
                stage_times.setdefault(stage, []).append(ms)
            stage_times.setdefault("TOTAL", []).append(sum(times.values()))
    rows = [(stage, f"{np.median(v):.1f}", f"{np.min(v):.1f}", len(v))
            for stage, v in stage_times.items()]
    h, w = cv2.imread(os.path.join(SYNTHETIC_DIR, "aligned.jpg")).shape[:2]
    print_table(f"STAGE TIMING ({w}x{h} input, 7 detected cases x {runs} runs)",
                ["Stage", "Median (ms)", "Min (ms)", "Samples"], rows)


def print_table(title, headers, rows):
    print(f"\n=== {title} ===")
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)] if rows else [len(h) for h in headers]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(x).ljust(w) for x, w in zip(row, widths)))


if __name__ == "__main__":
    test_synthetic()
    test_timing()
    test_real_photos()
    test_invalid_inputs()
