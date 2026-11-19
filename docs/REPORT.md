# SmartScan: An Automated Document Scanning and Image Enhancement System Using Computer Vision

**Course:** Image and Video Processing Lab (CSET344)
**Team:** _[Name 1 (Roll No.)], [Name 2 (Roll No.)], [Name 3 (Roll No.)]_
**Guide:** _[Faculty name]_

> **Note to the team:** All numbers in Section 7 come from `python test_pipeline.py`. Its full output is saved in `results/test_output.txt`. Those tests used **synthetic** photographs, for the reason explained in 7.1. Before submitting, add your own phone photos to `sample_images/`, re-run the script, and add those observations to Section 7.4. Remove this note before submission.

---

## 1. Abstract

Photographs of paper documents taken with a phone are usually tilted, include background clutter, and suffer from shadows and poor contrast. This makes them unsuitable for printing, sharing or record keeping. SmartScan is a desktop web application that converts such a photograph into a cropped, perspective-corrected and enhanced digital document using only classical image-processing techniques.

The system resizes and denoises the image (grayscale conversion, Gaussian and median filtering), removes text detail with morphological closing, and finds the page boundary with Canny edge detection, contour analysis and Douglas–Peucker polygon approximation, with Otsu thresholding as a fallback. The four detected corners are refined to sub-pixel accuracy, ordered, and used to compute a homography that warps the page into a flat rectangle. Four scan modes are offered: colour, grayscale, black-and-white (adaptive thresholding with morphological closing) and enhanced (background-division shadow removal with CLAHE). Histogram equalization, erosion, dilation, opening and closing are also available for comparison. Results can be exported as PNG, JPG or an aspect-preserving PDF.

On eight synthetic test photographs with known ground truth, the document was detected in seven. Mean corner error was 0.4–1.3 px in six cases and 11.7 px on a cluttered background. The median processing time was 116 ms per 1200×1600 image.

## 2. Introduction & Problem Statement

### 2.1 Introduction

Students, offices and small businesses often need digital copies of paper documents such as assignments, receipts, forms and ID proofs. A flatbed scanner gives clean results but is not always available. A phone camera is always available, but a raw photograph is a poor substitute for a scan:

- the page appears as a **trapezium**, not a rectangle, because of camera perspective;
- the **background** (table, hands, other objects) is included;
- **uneven lighting and shadows** make parts of the page darker;
- **low light** gives a dark, noisy image with low contrast.

### 2.2 Problem Statement

Design and implement a system that takes an ordinary photograph of a single physical document and automatically:

1. locates the document's four corners;
2. removes the background and corrects perspective distortion;
3. improves readability through enhancement;
4. exports the result as an image or PDF.

It must use classical image-processing algorithms, run on a normal laptop without a GPU or internet connection, and handle invalid input and detection failures gracefully.

## 3. Review of Existing Systems & Feasibility

### 3.1 Existing systems

| System | Approach | Limitations relevant to this project |
|---|---|---|
| Flatbed / sheet-fed scanners | Dedicated hardware | Costly, not portable, slow for one-off use |
| Mobile scanning apps (e.g. CamScanner [14], Adobe Scan [15], Microsoft Lens [16]) | Automatic edge detection and enhancement on the phone | Closed-source, so the algorithms cannot be studied; some features need an account or a subscription; documents may be uploaded to cloud services |
| Manual editing (crop and rotate in an image editor) | User crops by hand | Rotation and crop cannot remove perspective distortion; slow; no enhancement for shadows |
| Deep-learning document detectors (research) | CNNs predict corners or edges | Need training data and more computation; harder to explain and verify in a lab setting |

### 3.2 Gap

There is a need for a transparent, offline tool whose every processing stage can be **seen, compared and explained**: an educational and practical scanner rather than a black box.

### 3.3 Feasibility

- **Technical:** All required operations are available in OpenCV [9]. Measured processing takes about 0.1 s per image on a laptop CPU (Section 7.3).
- **Economic:** Only free, open-source software is used (Python, OpenCV, NumPy, Streamlit, Pillow, Matplotlib). No GPU or paid API is needed.
- **Operational:** The user only uploads a photo and picks a mode. It runs locally in a browser through Streamlit.
- **Schedule:** Classical algorithms need no dataset collection or model training, so the project fits a semester.

## 4. Objectives & Methodology

### 4.1 Objectives

1. Accept JPG/PNG uploads and reject invalid, empty, corrupted or tiny files with clear messages.
2. Automatically detect the four corners of the document, and allow manual correction when detection fails or is wrong.
3. Correct perspective so the output is a flat rectangular page.
4. Provide colour, grayscale, black-and-white and enhanced scan modes, plus individual techniques for comparison.
5. Show every processing stage side by side.
6. Measure processing time per stage, and compute MSE, PSNR and SSIM when a valid reference exists.
7. Export as PNG, JPG and aspect-preserving PDF.

### 4.2 Methodology (block diagram)

```
Photo ─► Validate/Decode ─► Pre-process ─► Detect boundary ─► Refine corners ─► Order points
                                (resize, gray,   (closing, Canny/Otsu,   (cornerSubPix)
                                 Gaussian,        contours, approxPolyDP)        │
                                 median)                │ fail                   ▼
                                                        └─► Manual corners ─► Perspective warp
                                                                                 │
     Export (PNG/JPG/PDF) ◄── Comparison & Evaluation ◄── Enhancement / Scan modes ◄┘
```

_Figure 1: Block diagram. Redraw this in draw.io or PowerPoint for the final report._

### 4.3 Development process

The project was built in phases: environment setup → core processing and detection → perspective correction → enhancement → Streamlit interface → evaluation → integrated testing. Each module was tested on its own before integration (Section 6.3).

## 5. Algorithms / Techniques Used

### 5.1 Pre-processing

- **Resizing (`cv2.resize`, INTER_AREA):** Detection runs on a copy that is 500 px tall. This reduces computation and suppresses fine texture. The scale ratio *r* is stored, and the detected corners are multiplied by *r*.
- **Grayscale conversion:** Y = 0.299R + 0.587G + 0.114B. Boundary detection depends on intensity changes, not colour.
- **Gaussian blur (5×5):** Convolution with a 2-D Gaussian kernel. It is a low-pass filter that suppresses noise which would otherwise produce false gradients.
- **Median blur (5×5):** A non-linear filter that replaces each pixel with the median of its neighbourhood. It removes impulse noise while preserving edges.

### 5.2 Document boundary detection

1. **Morphological closing (9×9, 2 iterations)** on the grayscale image. Closing is dilation followed by erosion. Dilation takes the local maximum, so thin dark text strokes on the bright page are removed, leaving a plain page region.
2. **Canny edge detection [1]:** gradient computation, non-maximum suppression, double thresholding and hysteresis. Thresholds are first chosen automatically as (1 ± 0.33) × median intensity. If that fails, the fixed pairs (75, 200) and (30, 100) are tried.
3. **Dilation (3×3)** of the edge map closes small gaps in the outline.
4. **Contour extraction [2]** with `cv2.findContours` (RETR_LIST, CHAIN_APPROX_SIMPLE). Contours are sorted by area.
5. **Filtering:** contours larger than 98% of the image (the photo frame) are skipped. The search stops below 15% of the image area.
6. **Polygon approximation (Ramer–Douglas–Peucker [3], [4])** with ε = 0.02 × perimeter. The first contour that reduces to exactly **4 vertices** and is **convex** is accepted as the document.
7. **Fallback: Otsu thresholding [5].** The global threshold that minimizes within-class variance separates a bright page from a dark background, and the same contour search is applied.
8. **Corner refinement:** `cv2.cornerSubPix` [9] runs on the full-resolution grayscale image, with a window proportional to the resize ratio. It iteratively moves each corner to the point where the image gradients in the window are orthogonal to the vectors from the corner, giving sub-pixel accuracy. A point that moves farther than the window size is rejected and the original kept.

If no quadrilateral is found, the user is told why this may have happened and can place the corners with sliders or use the whole image.

### 5.3 Perspective transformation

A plane viewed by a camera is related to its frontal view by a **homography** [6]:

$$\begin{bmatrix}x'w\\y'w\\w\end{bmatrix}=\begin{bmatrix}h_{11}&h_{12}&h_{13}\\h_{21}&h_{22}&h_{23}\\h_{31}&h_{32}&1\end{bmatrix}\begin{bmatrix}x\\y\\1\end{bmatrix}$$

H has 8 unknowns. Each point correspondence gives two equations, so four corners determine H, which is solved by `cv2.getPerspectiveTransform`.

- **Point ordering:** The angles of the corners around their centroid (`atan2`) give a clockwise order in image coordinates. The list is then rotated so the corner with minimum x + y is first (top-left). Unlike the common sum/difference rule, this cannot select the same point twice when the page is rotated by about 45°.
- **Output size:** width = max(|TR − TL|, |BR − BL|) and height = max(|BL − TL|, |BR − TR|).
- **Warping:** `cv2.warpPerspective` maps each output pixel back through H⁻¹ and bilinearly interpolates the full-resolution photo.

### 5.4 Enhancement techniques

| Technique | OpenCV function | Idea |
|---|---|---|
| Grayscale | `cvtColor` | Single intensity channel |
| Gaussian filtering | `GaussianBlur` | Noise suppression |
| Adaptive thresholding | `adaptiveThreshold` (Gaussian, block 21, C 10 by default) | Per-pixel threshold T(x, y) = Gaussian-weighted local mean − C, which tolerates uneven illumination |
| Histogram equalization | `equalizeHist` | Map intensities through the normalized CDF to spread them over 0–255 |
| CLAHE [7], [8] | `createCLAHE` (clip 2.0, 8×8 tiles) | Equalize each tile with a clipped histogram, then interpolate between tiles; limits noise amplification |
| Erosion / Dilation [10] | `erode` / `dilate` | Local min / max: white shrinks / grows (text thickens / thins) |
| Opening / Closing [10] | `morphologyEx` | Opening removes small white specks; closing removes small black specks on a white page |
| Illumination normalization | `dilate` 7×7 → `medianBlur` 21 → `divide` | Estimate the text-free background and divide by it to flatten shadows |

**Scan modes:**
- **Colour:** the warped image.
- **Grayscale:** the grayscale conversion.
- **Black & White:** Gaussian 3×3 → adaptive threshold → closing.
- **Enhanced:** per-channel illumination normalization → CLAHE on the L channel of LAB, so colours do not shift.

### 5.5 Evaluation metrics

- **MSE** = (1/MN) Σ (I − K)².
- **PSNR** = 10 log₁₀(255² / MSE) dB.
- **SSIM [11]** = ((2μₓμᵧ + C₁)(2σₓᵧ + C₂)) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂)), with an 11×11 Gaussian window (σ = 1.5), C₁ = (0.01·255)² and C₂ = (0.03·255)², averaged over the image. It is implemented from the formula with OpenCV and NumPy.
- **Mean corner error:** the average Euclidean distance (pixels) between detected and true corners, after ordering.
- **Processing time:** `time.perf_counter()` around each stage.

These are full-reference metrics, so they are computed **only against a reference of the same document**: the known clean page in synthetic tests, or a user-supplied reference scan in the app.

## 6. Implementation

### 6.1 Tools

Python 3.10.11, OpenCV 4.10.0, NumPy 1.26.4, Streamlit 1.39.0, Pillow 10.4.0, Matplotlib 3.9.2, on Windows 11 using a VS Code virtual environment. The versions are pinned in `requirements.txt`.

### 6.2 Project structure

```
app.py                         Streamlit user interface (connects all modules)
modules/preprocessing.py       load_image_from_bytes, resize_image, preprocess_image, prepare_for_detection
modules/document_detection.py  remove_text_details, find_edges, auto_canny_thresholds,
                               find_quadrilateral, detect_document, refine_corners, draw_boundary
modules/perspective_transform.py  order_points, distance, four_point_transform
modules/image_enhancement.py   to_grayscale, gaussian_filter, adaptive_threshold, histogram_equalization,
                               apply_clahe, erosion, dilation, opening, closing, normalize_illumination,
                               colour_scan, grayscale_scan, black_and_white_scan, enhanced_scan, all_techniques
modules/evaluation.py          mse, psnr, ssim, evaluate
modules/export_document.py     to_image_bytes, to_pdf_bytes
test_pipeline.py               synthetic test generation, detection/quality/timing tests, invalid-input tests
.streamlit/config.toml         UI colour theme
```

### 6.3 User interface

The interface has these parts (suggested screenshots in brackets):
- **Upload:** a file picker for JPG, JPEG and PNG, with file name, dimensions and size shown. _[Fig. 2: home screen with instructions]_
- **Sidebar:** usage instructions and settings (threshold block size, C, CLAHE clip limit, morphology kernel size).
- **Corner choice:** auto-detected, manual sliders, or full image. _[Fig. 3: manual corner adjustment]_
- **Detection tab:** original, denoised grayscale, edge map, boundary. _[Fig. 4]_
- **Scan Modes tab:** the four modes. _[Fig. 5: all four modes on one document]_
- **Enhancement Lab tab:** nine techniques in a grid; any one can be chosen as the final output. _[Fig. 6]_
- **Comparison tab:** the four stages side by side, plus a Matplotlib intensity histogram. _[Fig. 7]_
- **Evaluation tab:** a per-stage timing table, and an optional reference upload that produces an MSE/PSNR/SSIM table. _[Fig. 8]_
- **Download tab:** PNG, JPG or PDF. _[Fig. 9]_

### 6.4 Error handling

| Situation | Handling |
|---|---|
| Empty, corrupted or non-image file | `cv2.imdecode` returns None, and an error message is shown |
| Wrong extension | Blocked by the uploader's type filter |
| Image smaller than 100 px or larger than 50 MP | Error message |
| No document found | Error explaining likely causes; manual corners / full image offered |
| Non-convex manual corners | Warning |
| Region too small to warp | `ValueError` caught and shown |
| Export failure | Caught and shown |

### 6.5 How to run

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py          # opens http://localhost:8501
python test_pipeline.py       # regenerates test images, results/ and the tables below
```

## 7. Results & Discussion

### 7.1 Test setup

**Why synthetic tests:** MSE, PSNR, SSIM and corner error need a ground truth, meaning the exact clean document and the exact true corners. These are unknown for an ordinary phone photo. `test_pipeline.py` therefore draws a clean 850×1100 A4-shaped document (text, a table, a coloured block). It then perspective-warps it into a 1200×1600 "photo" at known corner positions, adds backgrounds and degradations, applies a 3×3 Gaussian camera blur, and saves it as JPEG. The eight cases are: aligned, tilted, strong perspective, shadow (brightness falls to 40% on one side), low light (30% brightness + Gaussian noise σ = 6), complex background (random shapes and lines), low-contrast background (light grey), and an occluded corner.

**Limitation:** synthetic images are simpler than real photos. They have no lens distortion, paper curl, real shadows or sensor-specific noise. The results below show the method works under controlled conditions and do not guarantee performance on real photos.

### 7.2 Detection accuracy

| Case | Detected | Method used | Mean corner error (px) |
|---|---|---|---|
| Aligned | Yes | Canny (auto) | 0.5 |
| Tilted | Yes | Canny (auto) | 0.4 |
| Perspective | Yes | Canny (auto) | 0.7 |
| Shadow | Yes | Canny (auto) | 0.6 |
| Low light | Yes | Canny (auto) | 0.6 |
| Complex background | Yes | Canny (auto) | 11.7 |
| Low-contrast background | Yes | Canny (30, 100) | 1.3 |
| Occluded corner | **No** | – | – |

_Table 1: Detection on synthetic photos (1200×1600)._

- **Sub-pixel refinement matters.** Before `cornerSubPix` was added, corner errors on the same images were 2.6–4.9 px. The dilated edge band lies slightly outside the true page edge, and this error is multiplied by the resize ratio (3.2). With refinement they dropped to 0.4–0.7 px. For the aligned case this raised grayscale SSIM from 0.769 to 0.902.
- **Complex background (11.7 px).** One side of the page bordered bright pink and lavender shapes with little contrast to white paper, so the contour cut slightly inside the page. The refinement step rejected the large correction, as designed. The output is usable but shows a measurable loss (SSIM 0.635 compared with 0.895 using the true corners).
- **Low-contrast background.** The automatic thresholds failed; the lower fixed thresholds (30, 100) succeeded.
- **Occluded corner.** The page outline no longer reduces to four vertices, so detection fails. The app reports this and offers manual corners.

### 7.3 Enhancement quality against the clean document

| Case | Output | MSE | PSNR (dB) | SSIM |
|---|---|---|---|---|
| Aligned | Grayscale | 810.3 | 19.04 | 0.9016 |
| | Black and White | 3677.1 | 12.48 | 0.7944 |
| | Enhanced | 1525.3 | 16.30 | 0.8789 |
| | Grayscale (true corners) | 744.2 | 19.41 | 0.9074 |
| Shadow | Grayscale | 7236.4 | 9.54 | 0.7952 |
| | Black and White | 3642.0 | 12.52 | 0.7950 |
| | **Enhanced** | **1525.3** | **16.30** | **0.8559** |
| | Grayscale (true corners) | 7174.1 | 9.57 | 0.8006 |
| Low light | Grayscale | 29114.2 | 3.49 | 0.4201 |
| | **Black and White** | **3427.9** | **12.78** | **0.8041** |
| | Enhanced | 2148.0 | 14.81 | 0.4891 |
| | Grayscale (true corners) | 29075.8 | 3.50 | 0.4229 |
| Complex background | Grayscale | 5258.1 | 10.92 | 0.6349 |
| | Grayscale (true corners) | 856.8 | 18.80 | 0.8948 |

_Table 2: Selected rows. The full table for all cases is in `results/test_output.txt`. "True corners" warps with the known corners, which separates detection error from imaging effects._

Discussion:

- **Shadow:** The enhanced mode cut MSE by 79% compared with grayscale (7236 → 1525) and raised SSIM from 0.795 to 0.856. Background division removes the illumination gradient, as intended.
- **Low light:** The grayscale output is very dark (PSNR 3.5 dB). The B&W mode gave the best SSIM (0.804) because adaptive thresholding depends on *local* contrast, not absolute brightness. The enhanced mode restored brightness (MSE fell from 29114 to 2148) but **amplified sensor noise**, so its SSIM (0.489) is barely above grayscale (0.420). This shows that a single metric can be misleading: MSE favours the enhanced output while SSIM penalizes the noise.
- **Under good lighting:** B&W has a lower PSNR/SSIM than grayscale even though it looks clean. Binarization turns smooth anti-aliased stroke edges into hard steps. It also turns the large filled blue block **hollow**, because adaptive thresholding only responds to local changes (see `results/aligned_3_bw.png`). The enhanced mode similarly lightens large solid colour areas, because the background estimate treats them as paper. B&W and Enhanced modes therefore suit text documents, not photos or large filled areas.
- **Absolute values:** These are modest (best PSNR about 19 dB) even with perfect corners, because the "photo" was blurred, JPEG-compressed and resampled twice. The metrics are useful for **comparing** methods on the same input, not as absolute quality scores.

### 7.4 Processing time

| Stage | Median (ms) | Min (ms) |
|---|---|---|
| Pre-processing | 6.7 | 5.5 |
| Detection | 2.9 | 2.2 |
| Corner refinement | 1.0 | 0.8 |
| Perspective warp | 4.8 | 3.3 |
| Grayscale mode | 0.4 | 0.2 |
| Black & White mode | 7.5 | 4.4 |
| Enhanced mode | 94.4 | 77.1 |
| **Total** | **116.0** | **98.5** |

_Table 3: 7 detected cases × 5 runs, 1200×1600 input, Windows 11 laptop CPU, with other applications open._

The Enhanced mode takes about 80% of the time because of the 21×21 median blur on three full-resolution channels. Detection itself takes only about 3 ms because it runs on the 500 px copy. Individual runs varied widely, up to several seconds, when other processes and OneDrive file syncing were active. That is why medians are reported. Times inside the Streamlit app are higher because the app also renders images.

### 7.5 Robustness tests

| Test | Result |
|---|---|
| Empty file | Rejected, with error message |
| Text file renamed to .jpg | Rejected, with error message |
| Truncated JPEG | Rejected, with error message |
| 60×50 image | Rejected ("too small") |
| Degenerate (tiny) warp region | `ValueError` caught |
| All 4 scan modes, 10 output choices, 3 export formats | No exceptions (automated Streamlit test) |
| PDF aspect ratio | Image 806×1102 (ratio 1.3672), PDF page 595.4×814.1 pt (ratio 1.3672), width 8.27" |

### 7.6 Real photographs

_[To be completed by the team. Put your photos in `sample_images/`, run `python test_pipeline.py`, and record which were detected (`results/real_*_boundary.jpg`) along with your visual observations of each scan mode. Include at least: flat page on a dark table, tilted shot, shadowed page, low light, busy background, and a case that fails. Only report what you actually observe.]_

_Suggested figures: input / boundary / warped / final for 3 cases (`results/*_1_boundary.jpg` … `*_4_enhanced.jpg`), and one failure case._

## 8. Conclusion & Future Scope

### 8.1 Conclusion

SmartScan shows that a complete document-scanning pipeline can be built from classical image-processing operations and runs in about 0.1 s on a laptop CPU. On controlled synthetic tests it detected the page in 7 of 8 cases with sub-pixel to roughly 1 px corner accuracy in 6 of them, corrected perspective, and removed shadows effectively. It also handled every invalid input tested without crashing. The tests also exposed real limitations: an occluded corner and low-contrast surroundings defeat contour-based detection, binarization destroys large filled regions, and illumination normalization amplifies noise in low light. The manual corner mode and the side-by-side comparisons let the user work around these cases.

### 8.2 Future scope

1. Detect the page from line segments (Hough transform) so that one hidden corner can be inferred from the other edges.
2. Denoise before enhancement in low light (for example non-local means, `cv2.fastNlMeansDenoising`).
3. Choose the adaptive threshold parameters automatically from the image content.
4. Multi-page scanning into one PDF.
5. OCR (for example Tesseract) to make the PDF searchable.
6. Dewarp curved pages (book spines) using text-line models.
7. Learned corner detection for highly cluttered backgrounds.
8. Real-time scanning from a webcam video stream.

## 9. References

[1] J. Canny, "A computational approach to edge detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. PAMI-8, no. 6, pp. 679–698, Nov. 1986.

[2] S. Suzuki and K. Abe, "Topological structural analysis of digitized binary images by border following," *Comput. Vis. Graph. Image Process.*, vol. 30, no. 1, pp. 32–46, 1985.

[3] U. Ramer, "An iterative procedure for the polygonal approximation of plane curves," *Comput. Graph. Image Process.*, vol. 1, no. 3, pp. 244–256, 1972.

[4] D. H. Douglas and T. K. Peucker, "Algorithms for the reduction of the number of points required to represent a digitized line or its caricature," *Cartographica*, vol. 10, no. 2, pp. 112–122, 1973.

[5] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1, pp. 62–66, Jan. 1979.

[6] R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2004.

[7] S. M. Pizer *et al.*, "Adaptive histogram equalization and its variations," *Comput. Vis. Graph. Image Process.*, vol. 39, no. 3, pp. 355–368, 1987.

[8] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in *Graphics Gems IV*, P. S. Heckbert, Ed. San Diego, CA, USA: Academic Press, 1994, pp. 474–485.

[9] G. Bradski, "The OpenCV library," *Dr. Dobb's J. Softw. Tools*, vol. 25, no. 11, pp. 120–125, 2000. Documentation: https://docs.opencv.org/4.10.0/

[10] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. New York, NY, USA: Pearson, 2018.

[11] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," *IEEE Trans. Image Process.*, vol. 13, no. 4, pp. 600–612, Apr. 2004.

[12] C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.

[13] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Comput. Sci. Eng.*, vol. 9, no. 3, pp. 90–95, 2007.

[14] CamScanner. [Online]. Available: https://www.camscanner.com

[15] Adobe Scan. [Online]. Available: https://www.adobe.com/acrobat/mobile/scanner-app.html

[16] Microsoft Lens. [Online]. Available: https://support.microsoft.com/office/microsoft-lens-for-android-ec124207-0fe3-4d5c-a7a4-1d3e4dd6eb2c

[17] Streamlit documentation. [Online]. Available: https://docs.streamlit.io

_Check the URLs [14]–[17] still resolve before submission._

## 10. Appendix A: Team Contribution

| Member | Contribution |
|---|---|
| _[Name 1]_ | _e.g. pre-processing and document detection modules, detection testing_ |
| _[Name 2]_ | _e.g. perspective transform and enhancement modules, comparison of techniques_ |
| _[Name 3]_ | _e.g. Streamlit UI, export, evaluation module, report_ |

_Fill in honestly. Each member should be able to explain their part (see `docs/VIVA_PREP.md`)._

## 11. Appendix B: Code Link & Self-Check

**Code link:** _[GitHub repository URL]_

| Self-check item | Status |
|---|---|
| Application runs with `streamlit run app.py` | ✅ Verified (server health check passed) |
| All modules import correctly | ✅ |
| Upload validation (empty / corrupt / tiny) | ✅ Tested |
| Automatic detection + failure message | ✅ Tested (7 of 8 synthetic cases detected) |
| Manual corner correction | ✅ Tested |
| Perspective correction | ✅ |
| Four scan modes + 9 lab techniques | ✅ |
| Before/after comparison + histogram | ✅ |
| MSE / PSNR / SSIM only with a valid reference | ✅ |
| Per-stage processing time | ✅ |
| PNG / JPG / PDF export, PDF aspect ratio preserved | ✅ Verified |
| Tested on real phone photographs | ⬜ **Team to complete** |
| Results in the report are actual measured values | ✅ (from `results/test_output.txt`) |
| References cited for all algorithms | ✅ |
