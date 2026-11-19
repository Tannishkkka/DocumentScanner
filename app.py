import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from modules.preprocessing import load_image_from_bytes, prepare_for_detection
from modules.document_detection import detect_document, draw_boundary, refine_corners
from modules.perspective_transform import four_point_transform, order_points
from modules.image_enhancement import (colour_scan, grayscale_scan, black_and_white_scan,
                                       enhanced_scan, all_techniques, to_grayscale)
from modules.evaluation import evaluate
from modules.export_document import to_image_bytes, to_pdf_bytes

MIN_SIDE = 100
MAX_PIXELS = 50_000_000
CORNER_NAMES = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]

st.set_page_config(page_title="SmartScan", page_icon="📄", layout="wide")


def show(image, caption):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption=caption, use_column_width=True)


def timed(times, label, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    times[label] = (time.perf_counter() - start) * 1000
    return result


def histogram_figure(before, after):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.hist(to_grayscale(before).ravel(), bins=256, range=(0, 255), alpha=0.6,
            label="Corrected (grayscale)", color="#64748b")
    ax.hist(to_grayscale(after).ravel(), bins=256, range=(0, 255), alpha=0.6,
            label="Final output", color="#2563eb")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Pixel count")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------- Header and sidebar ----------------
st.title("📄 SmartScan")
st.caption("Automated document scanning and image enhancement using computer vision")

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Upload a photo of a document (JPG/PNG).\n"
        "2. Check the detected boundary. Adjust corners if needed.\n"
        "3. Pick a scan mode or a technique from the Enhancement Lab.\n"
        "4. Compare stages, check evaluation, and download."
    )
    st.markdown("**Tips for good detection:** place the page on a darker, plain surface "
                "with all four corners visible.")
    st.divider()
    st.header("Enhancement settings")
    block_size = st.slider("Adaptive threshold block size", 3, 51, 21, step=2)
    c_value = st.slider("Adaptive threshold constant C", 0, 30, 10)
    clip_limit = st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, step=0.5)
    morph_size = st.slider("Morphology kernel size", 1, 7, 2)

# ---------------- Module 1: Upload ----------------
uploaded = st.file_uploader("Upload a document photograph", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a JPG, JPEG or PNG photograph of a document to begin. "
            "To scan another document later, just upload a new file here.")
    st.stop()

file_bytes = uploaded.getvalue()
image = load_image_from_bytes(file_bytes)
if image is None:
    st.error("This file could not be read as an image. It may be empty, corrupted, "
             "or not really a JPG/PNG file. Please upload a different file.")
    st.stop()

height, width = image.shape[:2]
if height < MIN_SIDE or width < MIN_SIDE:
    st.error(f"Image is too small ({width} x {height}). Each side must be at least {MIN_SIDE} pixels.")
    st.stop()
if height * width > MAX_PIXELS:
    st.error(f"Image is too large ({width} x {height}). Please upload an image under 50 megapixels.")
    st.stop()

st.write(f"**File:** {uploaded.name} &nbsp; | &nbsp; **Dimensions:** {width} x {height} pixels "
         f"&nbsp; | &nbsp; **Size:** {len(file_bytes) / 1024:.1f} KB")

times = {}

# ---------------- Modules 2 & 3: Pre-processing and detection ----------------
prep = timed(times, "Pre-processing", prepare_for_detection, image)
detected, edge_map, method = timed(times, "Document detection", detect_document, prep)
if detected is not None:
    detected = timed(times, "Corner refinement", refine_corners, image, detected, prep["ratio"])

file_key = f"{uploaded.name}_{uploaded.size}"
full_image = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)

if detected is not None:
    st.success(f"Document boundary detected (method: {method}).")
    options = ["Auto-detected corners", "Adjust corners manually", "Use full image"]
else:
    st.error("Could not find a four-sided document boundary in this photo. This usually happens "
             "when the page and background have similar colour, a corner is hidden, or the edges "
             "are blurred. You can place the corners manually below or use the full image.")
    options = ["Adjust corners manually", "Use full image"]

choice = st.radio("Document corners", options, horizontal=True, key=f"choice_{file_key}")

if choice == "Auto-detected corners":
    corners = order_points(detected)
elif choice == "Use full image":
    corners = full_image
else:
    if detected is not None:
        start = order_points(detected)
    else:
        margin_x, margin_y = width * 0.1, height * 0.1
        start = np.array([[margin_x, margin_y], [width - margin_x, margin_y],
                          [width - margin_x, height - margin_y], [margin_x, height - margin_y]])
    points = []
    with st.expander("Manual corner positions (pixels)", expanded=True):
        columns = st.columns(4)
        for i, column in enumerate(columns):
            with column:
                st.markdown(f"**{CORNER_NAMES[i]}**")
                x = st.slider("x", 0, width - 1, int(start[i][0]), key=f"{file_key}_x{i}")
                y = st.slider("y", 0, height - 1, int(start[i][1]), key=f"{file_key}_y{i}")
                points.append([x, y])
    corners = np.array(points, np.float32)
    if not cv2.isContourConvex(corners.reshape(-1, 1, 2).astype(np.int32)):
        st.warning("The four corners do not form a convex shape, so the result may look twisted. "
                   "Move them so they follow the page outline in order.")

boundary_image = draw_boundary(image, corners)

# ---------------- Module 4: Perspective transformation ----------------
try:
    warped, _ = timed(times, "Perspective transform", four_point_transform, image, corners)
except (ValueError, cv2.error) as error:
    st.error(f"Perspective correction failed: {error}")
    st.stop()

# ---------------- Modules 5 & 6: Enhancement and scan modes ----------------
modes = {
    "Original Colour": timed(times, "Mode: Colour", colour_scan, warped),
    "Grayscale": timed(times, "Mode: Grayscale", grayscale_scan, warped),
    "Black and White": timed(times, "Mode: Black and White", black_and_white_scan,
                             warped, block_size, c_value, morph_size),
    "Enhanced": timed(times, "Mode: Enhanced", enhanced_scan, warped, clip_limit),
}
techniques = timed(times, "All lab techniques", all_techniques,
                   warped, block_size, c_value, clip_limit, morph_size)

tab_detect, tab_modes, tab_lab, tab_compare, tab_eval, tab_download = st.tabs(
    ["🔍 Detection", "🖨️ Scan Modes", "🧪 Enhancement Lab", "↔️ Comparison",
     "📊 Evaluation", "⬇️ Download"])

with tab_detect:
    st.subheader("Pre-processing and boundary detection")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show(image, "Original photograph")
    with c2:
        show(prep["denoised"], "Grayscale + Gaussian + median blur")
    with c3:
        show(edge_map, f"Edge / threshold map ({method})")
    with c4:
        show(boundary_image, "Selected document boundary")
    st.caption("Detection runs on a copy resized to 500 px height for speed. The corners are "
               "scaled back, refined with cv2.cornerSubPix, and the warp uses the full-resolution photo.")

with tab_modes:
    st.subheader("Scanning modes")
    mode_name = st.radio("Choose a scan mode", list(modes), horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        show(warped, "Perspective-corrected document")
    with c2:
        show(modes[mode_name], f"{mode_name} scan")

with tab_lab:
    st.subheader("Compare individual enhancement techniques")
    st.caption("Settings for these are in the sidebar. Morphology is applied to the adaptive "
               "threshold (black-and-white) result.")
    names = list(techniques)
    for row_start in range(0, len(names), 3):
        columns = st.columns(3)
        for column, name in zip(columns, names[row_start:row_start + 3]):
            with column:
                show(techniques[name], name)
    lab_choice = st.selectbox("Use one of these as the final output instead of the scan mode?",
                              ["No, use the selected scan mode"] + names)

if lab_choice.startswith("No"):
    final_name, final_image = mode_name, modes[mode_name]
else:
    final_name, final_image = lab_choice, techniques[lab_choice]

with tab_compare:
    st.subheader(f"Before and after (final output: {final_name})")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show(image, "1. Original photograph")
    with c2:
        show(boundary_image, "2. Boundary detection")
    with c3:
        show(warped, "3. Perspective corrected")
    with c4:
        show(final_image, f"4. Final: {final_name}")
    st.markdown("**Intensity histogram**: shows how enhancement redistributes brightness.")
    figure = histogram_figure(warped, final_image)
    st.pyplot(figure)
    plt.close(figure)

with tab_eval:
    st.subheader("Processing time (this image)")
    total = sum(times.values())
    rows = [{"Stage": stage, "Time (ms)": f"{ms:.1f}"} for stage, ms in times.items()]
    rows.append({"Stage": "Total", "Time (ms)": f"{total:.1f}"})
    st.table(rows)
    st.caption("Measured with time.perf_counter() on this run. Times vary between runs and machines.")

    st.subheader("Quality metrics (needs a reference image)")
    st.markdown(
        "MSE, PSNR and SSIM only mean something when the output is compared with a **reference "
        "image of the same document**, for example a flatbed scan of the same page. "
        "Comparing with an unrelated image would give meaningless numbers, so no metrics are "
        "shown until you upload a reference.")
    reference_file = st.file_uploader("Upload a reference scan of the same document (optional)",
                                      type=["jpg", "jpeg", "png"], key="reference")
    if reference_file is not None:
        reference = load_image_from_bytes(reference_file.getvalue())
        if reference is None:
            st.error("The reference file could not be read as an image.")
        else:
            outputs = dict(modes)
            outputs[f"Final: {final_name}"] = final_image
            metric_rows = []
            for name, output in outputs.items():
                scores = evaluate(output, reference)
                metric_rows.append({"Output": name,
                                    "MSE": f"{scores['MSE']:.2f}",
                                    "PSNR (dB)": f"{scores['PSNR (dB)']:.2f}",
                                    "SSIM": f"{scores['SSIM']:.4f}"})
            st.table(metric_rows)
            st.caption("Images are converted to grayscale and the output is resized to the "
                       "reference size first. Small misalignments between the two lower the scores.")
    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **MSE**: average of squared pixel differences. 0 means identical; lower is better.\n"
            "- **PSNR** = 10·log10(255² / MSE), in decibels. Higher is better; it is infinite "
            "for identical images.\n"
            "- **SSIM**: compares local brightness, contrast and structure in small windows. "
            "Ranges up to 1 (identical) and follows human perception better than MSE.\n"
            "- **Processing time**: wall-clock time of each stage in milliseconds.")

with tab_download:
    st.subheader("Download the final document")
    show(final_image, f"Final output: {final_name}")
    file_format = st.radio("Format", ["PNG", "JPG", "PDF"], horizontal=True)
    base_name = uploaded.name.rsplit(".", 1)[0] + "_smartscan"
    try:
        if file_format == "PDF":
            data, mime = to_pdf_bytes(final_image), "application/pdf"
        else:
            data = to_image_bytes(final_image, file_format)
            mime = "image/png" if file_format == "PNG" else "image/jpeg"
        st.download_button(f"Download {file_format}", data,
                           file_name=f"{base_name}.{file_format.lower()}", mime=mime,
                           type="primary")
        h, w = final_image.shape[:2]
        st.caption(f"Output size: {w} x {h} pixels. The PDF page uses the same aspect ratio.")
    except (ValueError, OSError) as error:
        st.error(f"Export failed: {error}")
