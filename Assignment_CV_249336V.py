import numpy as np, cv2, matplotlib.pyplot as plt, os
from pathlib import Path

OUTDIR = Path("it5437_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def apply_piecewise_lut(img_gray, control_pts):
    xs = np.array([p[0] for p in control_pts], dtype=np.float32)
    ys = np.array([p[1] for p in control_pts], dtype=np.float32)
    lut = np.interp(np.arange(256, dtype=np.float32), xs, ys).astype(np.uint8)
    return cv2.LUT(img_gray, lut), lut

#Question 1
img = cv2.imread("data/emma.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image 'data/emma.jpg' not found or cannot be read.")
control_pts = [(0, 0), (50, 50), (50, 100), (150, 255), (150,150), (255,255)]
out, lut = apply_piecewise_lut(img, control_pts)

cv2.imwrite(str(OUTDIR/"q1_output.png"), out)
plt.figure(); plt.title("Q1 – Transform"); plt.xlabel("Input"); plt.ylabel("Output"); plt.plot(lut); plt.show(); savefig(OUTDIR/"q1_transform.png")
plt.figure(); plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(out, cmap="gray"); plt.title("Transformed"); plt.axis("off"); plt.show(); savefig(OUTDIR/"q1_compare.png")
