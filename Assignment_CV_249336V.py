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
#--------------------------------------------------------------------------
img = cv2.imread("data/emma.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image 'data/emma.jpg' not found or cannot be read.")
control_pts = [(0, 0), (50, 50), (50, 100), (150, 255), (150,150), (255,255)]
out, lut = apply_piecewise_lut(img, control_pts)

cv2.imwrite(str(OUTDIR/"q1_output.png"), out)
plt.figure(); plt.title("Q1 – Transform"); plt.xlabel("Input"); plt.ylabel("Output"); plt.plot(lut); plt.show(); savefig(OUTDIR/"q1_transform.png")
plt.figure(figsize=(10, 5)); plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(out, cmap="gray"); plt.title("Transformed"); plt.axis("off"); plt.show(); savefig(OUTDIR/"q1_compare.png")


#Question 2
#--------------------------------------------------------------------------
img = cv2.imread("data/brain_proton_density_slice.png", cv2.IMREAD_GRAYSCALE)

def midtone_stretch_lut(m1, m2, low_gain=0.6, mid_gain=1.8, high_gain=0.7):
    xs = [0, m1, (m1+m2)/2, m2, 255]
    ys = [0, int(m1*low_gain), int(255*0.5*mid_gain), int(min(255, m2*high_gain)), 255]
    return np.interp(np.arange(256), xs, ys).clip(0,255).astype(np.uint8)

lut_white = midtone_stretch_lut(160, 240, low_gain=0.1, mid_gain=3.0, high_gain=1.8)
lut_gray = midtone_stretch_lut(70, 150, low_gain=0.2, mid_gain=2.5, high_gain=0.4)

white = cv2.LUT(img, lut_white)
gray = cv2.LUT(img, lut_gray)

plt.figure(figsize=(10, 5)); plt.subplot(1,2,1); plt.title("Q2 – White Transform"); plt.plot(lut_white);
plt.subplot(1,2,2); plt.title("Q2 – Gray Transform"); plt.plot(lut_gray); plt.show(); savefig(OUTDIR/"q2_transforms_compare.png")

plt.figure(); plt.subplot(1,3,1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(white, cmap="gray"); plt.title("White Matter"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(gray, cmap="gray"); plt.title("Gray Matter"); plt.axis("off")
plt.show(); savefig(OUTDIR/"q2_compare.png")


#Question 3
#--------------------------------------------------------------------------
img = cv2.imread("data/highlights_and_shadows.jpg")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(lab)
gamma = 0.6
L_corr = np.clip(255.0 * ((L.astype(np.float32)/255.0) ** gamma), 0, 255).astype(np.uint8)
lab_corr = cv2.merge([L_corr, a, b])
img_corr = cv2.cvtColor(lab_corr, cv2.COLOR_LAB2BGR)
cv2.imwrite(str(OUTDIR/"q3_gamma_corrected.jpg"), img_corr)

plt.figure(figsize=(10, 5)); plt.subplot(1,2,1); plt.title("Q3 – L Histogram Before"); plt.hist(L.ravel(), bins=256, range=(0,255));
plt.subplot(1,2,2); plt.title("Q3 – L Histogram After"); plt.hist(L_corr.ravel(), bins=256, range=(0,255)); plt.show(); savefig(OUTDIR/"q3_hist_L_compare.png")

plt.figure(figsize=(10, 5)); plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(img_corr, cv2.COLOR_BGR2RGB)); plt.title("Q3 – Gamma Corrected Image"); plt.axis("off"); plt.show(); savefig(OUTDIR/"q3.png")


#Question 4
#--------------------------------------------------------------------------
#Part (a)
#-----------------------------------
img = cv2.imread("data/spider.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(H, cmap='gray'); plt.title('Q4 (a) - Hue Plane'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(S, cmap='gray'); plt.title('Q4 (a) - Saturation Plane'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(V, cmap='gray'); plt.title('Q4 (a) - Value Plane'); plt.axis('off')
plt.show()

#Part (b)
#-----------------------------------
sigma = 70.0
a = 0.7
x = np.arange(256, dtype=np.float32)
boost = a * 128.0 * np.exp(-((x - 128.0)**2)/(2.0*(sigma**2)))
f = np.minimum(x + boost, 255.0).astype(np.uint8)
S2 = cv2.LUT(S, f)
plt.figure(figsize=(8, 8)); plt.imshow(S2, cmap='gray'); plt.title('Q4 (b) - Transformed Saturation Plane'); plt.axis('off'); plt.show()

#Part (c)
#-----------------------------------
a = 0.9
print(f"Current value of 'a': {a}")

#Part (d)
#-----------------------------------
sigma = 70.0
a = 0.9 
x = np.arange(256, dtype=np.float32)
boost = a * 128.0 * np.exp(-((x - 128.0)**2)/(2.0*(sigma**2)))
f = np.minimum(x + boost, 255.0).astype(np.uint8)
S2 = cv2.LUT(S, f)
img2 = cv2.cvtColor(cv2.merge([H,S2,V]), cv2.COLOR_HSV2BGR)
plt.figure(); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); plt.title('Q4 (d) - Recombined Image'); plt.axis('off'); plt.show()

#Part (e)
#-----------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Q4 (e) - Original Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); plt.title('Q4 (e) - Vibrance Enhanced Image'); plt.axis('off')
plt.show()
plt.figure(); plt.plot(f); plt.title('Q4 (e) - Intensity Transformation (Saturation)'); plt.xlabel('Input Saturation'); plt.ylabel('Output Saturation'); plt.show()

#Question 5
#--------------------------------------------------------------------------
#Part (a)
#-----------------------------------
img = cv2.imread("data/jeniffer.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(H, cmap='gray'); plt.title('Hue Plane'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(S, cmap='gray'); plt.title('Saturation Plane'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(V, cmap='gray'); plt.title('Value Plane'); plt.axis('off')
plt.suptitle('Q5 (a) - HSV Planes')
plt.show()

#Part (b)
#-----------------------------------
_, mask = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(10, 6))
plt.plot(mask, cmap='gray'); plt.title('Q5 (b) - Thresholded Mask'); plt.axis('off') 
plt.show()

#Part (c)
#-----------------------------------
fg = cv2.bitwise_and(V, V, mask=mask)

hist_fg = cv2.calcHist([fg], [0], mask, [256], [0, 256])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(fg, cmap='gray'); plt.title('Q5 (c) - Foreground Only'); plt.axis('off')
plt.subplot(1, 2, 2); plt.plot(hist_fg); plt.title('Q5 (c) - Foreground Histogram'); plt.xlabel('Pixel Intensity'); plt.ylabel('Frequency')
plt.show()

#Part (d)
#-----------------------------------
hist_fg_cumsum = np.cumsum(hist_fg)

plt.figure(figsize=(8, 5))
plt.plot(hist_fg_cumsum); plt.title('Q5 (d) - Cumulative Sum of Foreground Histogram'); plt.xlabel('Pixel Intensity'); plt.ylabel('Cumulative Frequency')
plt.show()

#Part (e)
#-----------------------------------
cdf_min = hist_fg_cumsum[hist_fg_cumsum > 0].min()
total_fg_pixels = hist_fg.sum()
equalization_lut = np.zeros(256, dtype=np.uint8)
equalization_lut[hist_fg_cumsum > cdf_min] = np.round(((hist_fg_cumsum[hist_fg_cumsum > cdf_min] - cdf_min) / (total_fg_pixels - cdf_min)) * 255).astype(np.uint8)
eq_fg = cv2.LUT(fg, equalization_lut)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(fg, cmap='gray'); plt.title('Q5 (e) - Original Foreground'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(eq_fg, cmap='gray'); plt.title('Q5 (e) - Equalized Foreground'); plt.axis('off')
plt.show()

#Part (f)
#-----------------------------------
background = cv2.bitwise_and(V, V, mask=cv2.bitwise_not(mask))
V2 = background + eq_fg
img2 = cv2.cvtColor(cv2.merge([H, S, V2]), cv2.COLOR_HSV2BGR)

plt.figure(figsize=(12, 6)) # Increased figure size
plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Q5 (f) - Original Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); plt.title('Q5 (f) - Final Result'); plt.axis('off')
plt.show()