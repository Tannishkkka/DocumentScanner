# SmartScan: Presentation and Viva Preparation

## 1. Presentation Plan (8–10 minutes)

| # | Slide | Time | What to say |
|---|---|---|---|
| 1 | Title | 0:20 | SmartScan: An Automated Document Scanning and Image Enhancement System Using Computer Vision. Team names, CSET344. |
| 2 | Introduction | 0:45 | People photograph documents with phones. These photos are tilted, have background clutter, shadows and poor contrast. They are hard to read, print or submit. |
| 3 | Problem statement | 0:45 | Turn an ordinary photo of a paper into a cropped, flat, clean digital document, automatically, using classical image processing on a normal laptop (no GPU, no internet). |
| 4 | Existing systems | 0:45 | Flatbed scanners (costly, not portable), mobile apps like CamScanner / Adobe Scan / Microsoft Lens (closed-source, ads or watermarks, data leaves the device), manual cropping in image editors (slow, no perspective correction). |
| 5 | Proposed methodology / block diagram | 1:00 | Walk through the block diagram below. |
| 6 | Detection algorithms | 1:15 | Grayscale, Gaussian blur, closing to remove text, Canny, contours, approxPolyDP, largest convex quadrilateral. Otsu backup. |
| 7 | Perspective transformation | 1:00 | Point ordering, output size, homography with getPerspectiveTransform, warpPerspective. |
| 8 | Enhancement | 1:00 | Adaptive threshold, histogram equalization, CLAHE, morphology, illumination normalization. The four scan modes. |
| 9 | Live demo | 2:00 | Upload, detection tab, adjust a corner, scan modes, lab, comparison, download PDF. |
| 10 | Results | 0:45 | The detection table and quality table from `docs/REPORT.md` Section 7. Be honest about the failure cases. |
| 11 | Conclusion & future scope | 0:30 | What works, what fails, what can be added. |
| 12 | Individual contribution | 0:20 | Who did what. |

### Block diagram

```
 Photo (JPG/PNG)
      │
      ▼
 [Validate & decode] ── invalid ──► error message
      │
      ▼
 [Pre-processing] resize to 500 px height → grayscale → Gaussian blur → median blur
      │
      ▼
 [Detection] closing (remove text) → Canny / Otsu → contours → sort by area
            → approxPolyDP → first convex 4-point contour
      │                         │
      │ found                   │ not found
      ▼                         ▼
 4 corners (scaled back,   manual corners / full image
 refined by cornerSubPix)
      │
      ▼
 [Perspective transform] order points → compute width, height
                         → getPerspectiveTransform → warpPerspective
      │
      ▼
 [Enhancement] Colour | Grayscale | B&W (adaptive threshold + closing) | Enhanced (shadow removal + CLAHE)
               + Lab: Gaussian, Hist. Eq., CLAHE, Erosion, Dilation, Opening, Closing
      │
      ▼
 [Comparison + Evaluation] stages side by side, histogram, timings, MSE/PSNR/SSIM vs reference
      │
      ▼
 [Export] PNG / JPG / PDF
```

### Demo script (2 minutes)

1. `streamlit run app.py`, upload a tilted photo. Point at the dimensions line.
2. **Detection tab**: "This is the blurred grayscale, this is the Canny edge map, and the green outline is the four corners we found."
3. Switch to **Adjust corners manually**, move one slider, show the outline move. "If detection is wrong the user can fix it."
4. **Scan Modes**: click through all four modes.
5. **Enhancement Lab**: point out hist. eq. vs CLAHE, and erosion vs dilation.
6. **Comparison**: the four stages and the histogram.
7. **Evaluation**: the timings. Explain why metrics need a reference.
8. **Download**: download the PDF and open it.
9. Keep a backup: if the live photo fails, use `sample_images/synthetic/tilted.jpg`. Also show the failure handling with `low_contrast_background.jpg` and explain why it fails.

---

## 2. Viva Questions and Answers

### Image processing fundamentals

**Q: What is a digital image?**
A 2-D array of pixels. A grayscale image stores one intensity (0 = black to 255 = white) per pixel. A colour image stores three values per pixel. OpenCV stores them in **BGR** order, not RGB.

**Q: Why convert to grayscale before detection?**
Edges and shapes depend on brightness changes, not colour. One channel instead of three is faster and simpler. The formula is Y = 0.299R + 0.587G + 0.114B.

**Q: Why resize to 500 px before detection?**
A 12 MP phone photo has about 12 million pixels. Finding the page outline does not need that detail, so detection on a small copy is much faster and less sensitive to fine texture. We multiply the corners by the resize ratio and warp the **full-resolution** image, so no quality is lost.

**Q: Why does the code use `cv2.imdecode` and not `cv2.imread`?**
Streamlit gives the uploaded file as bytes in memory, not as a path on disk. `imdecode` decodes bytes. If the bytes are not a valid image it returns `None`, which we use to show an error.

### Gaussian blur

**Q: What does Gaussian blur do?**
It replaces each pixel with a weighted average of its neighbours. The weights follow a 2-D Gaussian (bell curve), so nearer pixels count more. It is a low-pass filter: it removes high-frequency noise.

**Q: Why blur before Canny?**
Canny uses image gradients, and noise creates many false gradients. Blurring removes noise so that only strong, real edges remain.

**Q: What do `(5, 5)` and `0` mean in `cv2.GaussianBlur(gray, (5, 5), 0)`?**
`(5, 5)` is the kernel size and must be odd. `0` tells OpenCV to compute sigma from the kernel size.

**Q: Why also use a median blur?**
A median filter replaces each pixel with the median of its neighbourhood. It is very good at removing isolated noisy pixels (salt-and-pepper noise) while keeping edges sharp, which a mean filter does not.

### Canny edge detection

**Q: Explain the steps of Canny.**
1. Gaussian smoothing.
2. Gradient magnitude and direction using Sobel filters.
3. **Non-maximum suppression**: keep only pixels that are a local maximum along the gradient direction, which gives thin 1-pixel edges.
4. **Double threshold**: pixels above `high` are strong edges, pixels between `low` and `high` are weak, and pixels below `low` are discarded.
5. **Hysteresis**: weak pixels are kept only if they are connected to strong ones.

**Q: How did you choose the thresholds?**
We try several. The first is "auto" thresholds at ±33% around the median brightness, so they adapt to dark or bright photos. Then fixed pairs (75, 200) and (30, 100). If all fail, we use Otsu thresholding.

**Q: Why dilate the edge map?**
Edges of the page can have small gaps. Dilation thickens them so the outline becomes one closed contour.

**Q: Why apply morphological closing before Canny?**
Closing (dilate, then erode) with a 9×9 kernel on the grayscale image removes thin dark strokes, i.e. the text. The page becomes a plain bright region, so Canny finds mainly its outline.

### Contours and polygon approximation

**Q: What is a contour?**
A curve joining continuous boundary points of the same intensity. `cv2.findContours` traces the boundaries of white regions in a binary image.

**Q: `RETR_LIST` vs `RETR_EXTERNAL`?**
`RETR_EXTERNAL` returns only outermost contours. `RETR_LIST` returns all contours without hierarchy. We use `RETR_LIST` because the page outline can sit inside another contour, for example a table edge.

**Q: What does `CHAIN_APPROX_SIMPLE` do?**
It stores only the end points of straight segments instead of every boundary pixel, which saves memory.

**Q: How do you pick the document contour?**
Sort contours by area, largest first. Skip any contour bigger than 98% of the image (that is the photo border). Stop once contours are smaller than 15% of the image. For each remaining contour, approximate it with `approxPolyDP(contour, 0.02 * perimeter, True)` and accept the first one with **exactly 4 points** that is **convex**.

**Q: What is `approxPolyDP`?**
The Ramer–Douglas–Peucker algorithm. It simplifies a curve into fewer points so that no original point is farther than epsilon from the simplified shape. Epsilon = 2% of the perimeter. A rough rectangle simplifies to 4 points.

**Q: Why check convexity?**
A real sheet of paper seen by a camera is always a convex quadrilateral. A concave 4-point shape is a false detection.

**Q: When does detection fail?**
When the page and background have similar brightness, a corner is hidden, the page is curled or folded, or the background has strong rectangular shapes larger than the page. The app then shows an error and offers manual corners or the full image.

**Q: What does `cornerSubPix` do and why did you add it?**
Detection runs on a 500 px copy, and the dilated edge band sits slightly outside the real page edge. After scaling back by about 3×, the corners were 2.6–4.9 px off in our tests. `cornerSubPix` looks at a small window around each corner in the full-resolution image. It moves the point until the gradient at every nearby pixel is perpendicular to the line from the corner to that pixel, which is exactly true at a real corner. This reduced the error to 0.4–0.7 px and raised SSIM from 0.77 to 0.90 on the aligned test. If a point jumps too far, for example when pulled by nearby text, we keep the original.

### Perspective transformation

**Q: What is a perspective transform (homography)?**
A mapping between two planes, written as a 3×3 matrix H with 8 unknowns (the last value is fixed to 1). A point (x, y) maps to (x', y') = ((h11·x + h12·y + h13)/w, (h21·x + h22·y + h23)/w), where w = h31·x + h32·y + 1. Unlike an affine transform, it can turn a trapezium into a rectangle, which is how a camera's view of a flat page is distorted.

**Q: Why exactly 4 points?**
H has 8 unknowns, and each point pair gives 2 equations (one for x, one for y). So 4 pairs give 8 equations. `cv2.getPerspectiveTransform` solves this linear system. No 3 of the points may be collinear.

**Q: Why order the points?**
The source and destination points must correspond: top-left to (0, 0), top-right to (w−1, 0), and so on. If the order is wrong, the image comes out flipped or twisted.

**Q: How does your `order_points` work?**
Compute the centre of the 4 points. Sort the points by their angle around the centre using `arctan2`. Because the image y-axis points down, sorting by angle gives clockwise order. Then rotate the list so the point with the smallest x + y (nearest the top-left of the image) comes first. This handles strongly rotated pages better than the common "sum/difference" trick, which can give the same point twice when the page is rotated near 45°.

**Q: How is the output size decided?**
Width = the longer of the top and bottom edges. Height = the longer of the left and right edges (Euclidean distance). This keeps as much detail as possible.

**Q: What does `warpPerspective` do internally?**
For each output pixel, it applies the inverse of H to find the matching location in the source image and interpolates a value there (bilinear by default). Working backwards like this avoids holes in the output.

### Adaptive thresholding

**Q: Global vs adaptive threshold?**
A global threshold uses one value T for the whole image. With a shadow, the dark half falls below T and turns black. An adaptive threshold computes a separate T for each pixel from its neighbourhood, so it handles uneven lighting.

**Q: Explain `cv2.adaptiveThreshold(gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 10)`.**
For each pixel, T = (Gaussian-weighted mean of the 21×21 neighbourhood) − 10. If the pixel is above T it becomes 255 (white), otherwise 0 (black). Block size must be odd. A larger C means fewer black pixels (less noise, but thin text can disappear).

**Q: What is Otsu's method?**
It chooses the global threshold that minimizes the variance within the two classes (equivalently, maximizes the variance between them) of the histogram. We use it only as a backup for detection, to separate a bright page from a dark background.

### Histogram equalization and CLAHE

**Q: What is a histogram?**
A count of how many pixels have each intensity 0–255.

**Q: How does histogram equalization work?**
It maps each intensity through the normalized cumulative histogram (CDF): new = round(255 · CDF(old)). This spreads the intensities over the full range and increases global contrast.

**Q: Problem with plain equalization?**
It is global, so it can over-amplify noise and wash out regions. On a document that is mostly white paper, it can make the background grey and noisy. You can see this in the Enhancement Lab.

**Q: What is CLAHE?**
Contrast Limited Adaptive Histogram Equalization. The image is split into tiles (8×8 here) and each tile is equalized separately. Each tile's histogram is **clipped** at the clip limit and the excess is redistributed, which limits how much noise gets amplified. Neighbouring tiles are blended by bilinear interpolation to avoid visible tile borders.

**Q: Why apply CLAHE on the L channel of LAB?**
L holds lightness only, and A and B hold colour. Changing only L improves contrast without shifting colours.

### Morphological operations

**Q: Define erosion and dilation.**
With a structuring element (kernel), **erosion** sets a pixel to the minimum in its neighbourhood, so white regions shrink. **Dilation** uses the maximum, so white regions grow.

**Q: Opening vs closing?**
Opening = erosion then dilation. It removes small white objects. Closing = dilation then erosion. It fills small black holes and dots.

**Q: On a black-text, white-background scan, which removes black specks?**
**Closing.** This is a common trick question. OpenCV morphology acts on white pixels. Black specks on the white background are "small black holes" in the white, so closing fills them. Erosion grows black areas, so text looks **thicker**. Dilation makes text **thinner**.

**Q: How does your "Enhanced" mode remove shadows?**
For each colour channel, we estimate the background by dilation (removes dark text) followed by a large median blur (smooths it). Then we divide the channel by this background and scale by 255. Where the page was in shadow, the background is also dark, so the division brings it back up to white. This is a form of illumination normalization. After that, CLAHE is applied on the L channel.

### Evaluation metrics

**Q: Define MSE.**
MSE = (1 / MN) · Σ (I(x, y) − K(x, y))². It is the average squared pixel difference. 0 means identical images.

**Q: Define PSNR.**
PSNR = 10 · log10(MAX² / MSE), with MAX = 255 for 8-bit images. It is measured in dB, and higher is better. It is infinite for identical images.

**Q: Define SSIM.**
SSIM(x, y) = ((2μxμy + C1)(2σxy + C2)) / ((μx² + μy² + C1)(σx² + σy² + C2)). It compares luminance (means), contrast (variances) and structure (covariance) in local windows. We use an 11×11 Gaussian window with σ = 1.5, C1 = (0.01·255)² and C2 = (0.03·255)², then average over the image. Its maximum is 1.

**Q: Why is SSIM better than PSNR?**
PSNR treats every pixel error equally. Humans notice structural changes more than small uniform brightness shifts. SSIM models this, so it agrees better with human judgement.

**Q: Why don't you show MSE/PSNR/SSIM for every upload?**
These are **full-reference** metrics, so they need a correct "ideal" image of the same document. Comparing the output with the original photo is meaningless, because the photo has a background and a different geometry. So the app computes metrics only when the user uploads a reference scan. In testing, we used synthetic photos made from a known clean document, which gives a true reference.

**Q: Why might the B&W output have a lower SSIM than expected even if it looks good?**
Binarization changes anti-aliased text edges to pure black or white, and the resizing and warping cause sub-pixel misalignment. SSIM is very sensitive to both.

### OpenCV functions used

| Function | Purpose |
|---|---|
| `cv2.imdecode` | decode uploaded bytes into an image |
| `cv2.resize` (INTER_AREA) | shrink for detection |
| `cv2.cvtColor` | BGR↔Gray, BGR↔RGB, BGR↔LAB |
| `cv2.GaussianBlur`, `cv2.medianBlur` | noise reduction |
| `cv2.morphologyEx` (CLOSE/OPEN), `cv2.erode`, `cv2.dilate` | morphology |
| `cv2.Canny` | edges |
| `cv2.threshold` (OTSU) | backup segmentation |
| `cv2.findContours`, `cv2.contourArea`, `cv2.arcLength`, `cv2.approxPolyDP`, `cv2.isContourConvex` | finding the page |
| `cv2.polylines`, `cv2.circle` | drawing the boundary |
| `cv2.getPerspectiveTransform`, `cv2.warpPerspective` | perspective correction |
| `cv2.adaptiveThreshold` | B&W scan |
| `cv2.equalizeHist`, `cv2.createCLAHE` | contrast |
| `cv2.divide` | illumination normalization |
| `cv2.imencode` | JPG/PNG export |

### General / design questions

**Q: Why not deep learning?**
The task can be solved with classical methods. They need no training data or GPU, run in milliseconds, and every step can be explained. Deep models (for example edge-detection networks like HED, or corner-regression CNNs) would be more robust on hard backgrounds. That is listed as future scope.

**Q: What are the limitations?**
- The page must be the largest 4-sided shape with reasonable contrast to the background.
- Curved or folded pages cannot be flattened by a single homography.
- A hidden corner breaks detection, and manual correction is needed.
- The adaptive threshold parameters may need tuning per photo.
- There is no OCR.

**Q: Why Streamlit?**
It makes a web UI directly from a Python script with no HTML or JavaScript, which keeps the focus on the image-processing part.
