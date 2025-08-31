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

#Question 2
img = cv2.imread("data/brain_proton_density_slice.png", cv2.IMREAD_GRAYSCALE)

def midtone_stretch_lut(m1, m2, low_gain=0.6, mid_gain=1.8, high_gain=0.7):
    xs = [0, m1, (m1+m2)/2, m2, 255]
    ys = [0, int(m1*low_gain), int(255*0.5*mid_gain), int(min(255, m2*high_gain)), 255]
    return np.interp(np.arange(256), xs, ys).clip(0,255).astype(np.uint8)

lut_white = midtone_stretch_lut(160, 240, low_gain=0.1, mid_gain=3.0, high_gain=1.8)
lut_gray = midtone_stretch_lut(70, 150, low_gain=0.2, mid_gain=2.5, high_gain=0.4)

white = cv2.LUT(img, lut_white)
gray = cv2.LUT(img, lut_gray)

plt.figure(); plt.title("Q2 – White Transform"); plt.plot(lut_white); plt.show(); savefig(OUTDIR/"q2_white_transform.png")
plt.figure(); plt.title("Q2 – Gray Transform"); plt.plot(lut_gray); plt.show(); savefig(OUTDIR/"q2_gray_transform.png")

plt.figure(); plt.subplot(1,3,1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(white, cmap="gray"); plt.title("White Matter"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(gray, cmap="gray"); plt.title("Gray Matter"); plt.axis("off")
plt.show(); savefig(OUTDIR/"q2_compare.png")

#Question 3
img = cv2.imread("data/highlights_and_shadows.jpg")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(lab)
gamma = 0.6
L_corr = np.clip(255.0 * ((L.astype(np.float32)/255.0) ** gamma), 0, 255).astype(np.uint8)
lab_corr = cv2.merge([L_corr, a, b])
img_corr = cv2.cvtColor(lab_corr, cv2.COLOR_LAB2BGR)
cv2.imwrite(str(OUTDIR/"q3_gamma_corrected.jpg"), img_corr)

plt.figure(); plt.title("Q3 – L Histogram Before"); plt.hist(L.ravel(), bins=256, range=(0,255)); plt.show(); savefig(OUTDIR/"q3_hist_L_before.png")
plt.figure(); plt.title("Q3 – L Histogram After"); plt.hist(L_corr.ravel(), bins=256, range=(0,255)); plt.show(); savefig(OUTDIR/"q3_hist_L_after.png")
plt.figure(); plt.title("Q3 – Gamma Corrected Image"); plt.imshow(cv2.cvtColor(img_corr, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()