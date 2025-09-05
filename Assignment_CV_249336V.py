import numpy as np, cv2, matplotlib.pyplot as plt, os

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

plt.figure(); plt.title("Q1 – Transform"); plt.xlabel("Input"); plt.ylabel("Output"); plt.plot(lut); plt.show();
plt.figure(figsize=(10, 5)); plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(out, cmap="gray"); plt.title("Transformed"); plt.axis("off"); plt.show()


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
plt.subplot(1,2,2); plt.title("Q2 – Gray Transform"); plt.plot(lut_gray); plt.show()

plt.figure(); plt.subplot(1,3,1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(white, cmap="gray"); plt.title("White Matter"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(gray, cmap="gray"); plt.title("Gray Matter"); plt.axis("off")
plt.show()


#Question 3
#--------------------------------------------------------------------------
img = cv2.imread("data/highlights_and_shadows.jpg")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(lab)
gamma = 0.6
L_corr = np.clip(255.0 * ((L.astype(np.float32)/255.0) ** gamma), 0, 255).astype(np.uint8)
lab_corr = cv2.merge([L_corr, a, b])
img_corr = cv2.cvtColor(lab_corr, cv2.COLOR_LAB2BGR)

plt.figure(figsize=(10, 5)); plt.subplot(1,2,1); plt.title("Q3 – L Histogram Before"); plt.hist(L.ravel(), bins=256, range=(0,255));
plt.subplot(1,2,2); plt.title("Q3 – L Histogram After"); plt.hist(L_corr.ravel(), bins=256, range=(0,255)); plt.show()

plt.figure(figsize=(10, 5)); plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(img_corr, cv2.COLOR_BGR2RGB)); plt.title("Q3 – Gamma Corrected Image"); plt.axis("off"); plt.show()


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
plt.imshow(mask, cmap='gray'); plt.title('Q5 (b) - Thresholded Mask'); plt.axis('off') 
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

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Q5 (f) - Original Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); plt.title('Q5 (f) - Final Result'); plt.axis('off')
plt.show()


#Question 6
#--------------------------------------------------------------------------
#Part (a)
#-----------------------------------
img = cv2.imread("data/einstein.png", cv2.IMREAD_GRAYSCALE)
assert img is not None, "Could not load data/einstein.png"

Kx = np.array([[ 1, 0, -1],
               [ 2, 0, -2],
               [ 1, 0, -1]], dtype=np.float32)
Ky = np.array([[ 1,  2,  1],
               [ 0,  0,  0],
               [-1, -2, -1]], dtype=np.float32)

Gx = cv2.filter2D(img, cv2.CV_32F, Kx)
Gy = cv2.filter2D(img, cv2.CV_32F, Ky)
mag = cv2.magnitude(Gx, Gy)
mag = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Einstein Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(mag, cmap="gray")
plt.title("Sobel (filter2D) – Gradient Magnitude")
plt.axis("off")
plt.show()


#Part (b)
#-----------------------------------
def conv2d_naive(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = cv2.copyMakeBorder(img, ph, ph, pw, pw, borderType=cv2.BORDER_REFLECT)
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            roi = padded[i:i+kh, j:j+kw].astype(np.float32)
            out[i, j] = np.sum(roi * kernel)
    return out

Gx_manual = conv2d_naive(img, Kx)
Gy_manual = conv2d_naive(img, Ky)
mag_manual = cv2.magnitude(Gx_manual, Gy_manual)
mag_manual = (mag_manual / (mag_manual.max() + 1e-8) * 255).astype(np.uint8)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Einstein Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(mag_manual, cmap="gray")
plt.title("Sobel (manual) – Gradient Magnitude")
plt.axis("off")
plt.show()


#Part (c)
#-----------------------------------
smooth = np.array([1, 2, 1], dtype=np.float32)      
deriv  = np.array([1, 0, -1], dtype=np.float32)     
smooth_col = smooth.reshape(-1, 1)   
deriv_row  = deriv.reshape(1, -1)   
deriv_col  = deriv.reshape(-1, 1)   
smooth_row = smooth.reshape(1, -1)  
Gx = cv2.sepFilter2D(img, ddepth=cv2.CV_32F, kernelX=deriv_row, kernelY=smooth_col)
Gy = cv2.sepFilter2D(img, ddepth=cv2.CV_32F, kernelX=smooth_row, kernelY=deriv_col)
mag_separable = cv2.magnitude(Gx, Gy)
mag_separable = (mag_separable / (mag_separable.max() + 1e-8) * 255).astype(np.uint8)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Einstein Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(mag_separable, cmap="gray")
plt.title("Sobel (separable) – Gradient Magnitude")
plt.axis("off")
plt.show()


plt.figure(figsize=(15, 8))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Einstein Image")
plt.axis("off")
plt.subplot(1, 4, 2)
plt.imshow(mag, cmap="gray")
plt.title("Sobel (filter2D) – Gradient Magnitude")
plt.axis("off")
plt.subplot(1, 4, 3)
plt.imshow(mag_manual, cmap="gray")
plt.title("Sobel (manual) – Gradient Magnitude")
plt.axis("off")
plt.subplot(1, 4, 4)
plt.imshow(mag_separable, cmap="gray")
plt.title("Sobel (separable) – Gradient Magnitude")
plt.axis("off")
plt.show()


#Question 7
#--------------------------------------------------------------------------
#Part (a) Nearest-neighbor interpolation
#-----------------------------------
pairs = [
    ("data/im01small.png", "data/im01.png"),
    ("data/im02small.png", "data/im02.png"),
    ("data/im03small.png", "data/im03.png"),
]

def zoom_nearest_to_size(img, H, W):
    h, w = img.shape[:2]
    ys = (np.linspace(0, h-1, H)).round().astype(int)
    xs = (np.linspace(0, w-1, W)).round().astype(int)
    return img[ys[:, None], xs[None, :]]

nearest_results, originals, smalls = [], [], []

for small_name, large_name in pairs:
    small = cv2.imread(f"{small_name}")
    large = cv2.imread(f"{large_name}")
    assert small is not None and large is not None, f"Missing: {small_name}, {large_name}"
    H, W = large.shape[:2]
    up_near = zoom_nearest_to_size(small, H, W)
    nearest_results.append(up_near); originals.append(large); smalls.append(small)

    plt.figure(figsize=(12,4))
    plt.suptitle(f"{small_name} → Nearest zoom vs {large_name}")
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)); plt.title("Small Input"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(cv2.cvtColor(up_near, cv2.COLOR_BGR2RGB)); plt.title("Zoomed (Nearest)"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(large, cv2.COLOR_BGR2RGB)); plt.title("Large Original"); plt.axis("off")
    plt.show()



# Part (b) Bilinear interpolation
#-----------------------------------
def zoom_bilinear_to_size(img, H, W):
    return cv2.resize(img, (W,H), interpolation=cv2.INTER_LINEAR)

bilinear_results = []

for (small_name, large_name), small, large in zip(pairs, smalls, originals):
    H, W = large.shape[:2]
    up_bilin = zoom_bilinear_to_size(small, H, W)
    bilinear_results.append(up_bilin)

    plt.figure(figsize=(12,4))
    plt.suptitle(f"{small_name} → Bilinear zoom vs {large_name}")
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)); plt.title("Small Input"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(cv2.cvtColor(up_bilin, cv2.COLOR_BGR2RGB)); plt.title("Zoomed (Bilinear)"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(large, cv2.COLOR_BGR2RGB)); plt.title("Large Original"); plt.axis("off")
    plt.show()

#Comparison of Nearest and Bilinear with Original
#-----------------------------------
def nssd(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.sum((a - b)**2) / (np.sum(b**2) + 1e-8))

names = ["im01", "im02", "im03"]

fig, ax = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle("Original vs Nearest vs Bilinear (small → zoomed to original size)", fontsize=14, y=0.95)

for r, (name, orig, near, bilin) in enumerate(zip(names, originals, nearest_results, bilinear_results)):
    ax[r,0].imshow(cv2.cvtColor(orig,  cv2.COLOR_BGR2RGB));  ax[r,0].set_title(f"{name} Large Original"); ax[r,0].axis("off")
    ax[r,1].imshow(cv2.cvtColor(near,  cv2.COLOR_BGR2RGB));  ax[r,1].set_title(f"{name} Nearest Zoom");   ax[r,1].axis("off")
    ax[r,2].imshow(cv2.cvtColor(bilin, cv2.COLOR_BGR2RGB));  ax[r,2].set_title(f"{name} Bilinear Zoom");  ax[r,2].axis("off")

plt.tight_layout()
plt.show()

print("=== Q7 Metrics (lower NSSD means closer to original) ===")
for name, orig, near, bilin in zip(names, originals, nearest_results, bilinear_results):
    H, W = orig.shape[:2]
    near = near[:H, :W]
    bilin = bilin[:H, :W]

    nssd_near  = nssd(near,  orig)
    nssd_bilin = nssd(bilin, orig)

    print(f"{name}: NSSD(nearest) = {nssd_near:.6f} | NSSD(bilinear) = {nssd_bilin:.6f}")



#Question 8
#--------------------------------------------------------------------------
#Part (a)
# -----------------------------------
img = cv2.imread("data/daisy.jpg")
h,w = img.shape[:2]
rect = (int(0.1*w), int(0.1*h), int(0.8*w), int(0.8*h))
mask = np.zeros((h,w), np.uint8)
bgdModel = np.zeros((1,65), np.float64); fgdModel = np.zeros((1,65), np.float64)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
fg = img * mask2[:,:,None]
mask_bg = cv2.bitwise_not(mask2)
bg = img * mask_bg[:,:,None]

plt.figure(figsize=(15, 5));
plt.subplot(1,3,1); plt.title("Q8 (a) – GrabCut Mask"); plt.imshow(mask2, cmap="gray"); plt.axis("off");
plt.subplot(1,3,2); plt.title("Q8 (a) – Foreground"); plt.imshow(cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)); plt.axis("off");
plt.subplot(1,3,3); plt.title("Q8 (a) – Background"); plt.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)); plt.axis("off");
plt.suptitle("Q8 (a) - GrabCut Segmentation Results")
plt.show();

#Part (b)
#------------------------------------
blurred = cv2.GaussianBlur(img, (31,31), 0)
enhanced = fg + blurred*(1-mask2)[:,:,None]
plt.figure(figsize=(10, 5));
plt.subplot(1,2,1); plt.title("Q8 (b) – Original"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis("off");
plt.subplot(1,2,2); plt.title("Q8 (b) – Enhanced Image"); plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)); plt.axis("off");
plt.suptitle("Q8 (b) - Enhanced Image with Blurred Background")
plt.show();


#Question 9
#--------------------------------------------------------------------------
#Part (a)
#------------------------------------
img = cv2.imread("data/rice.png", cv2.IMREAD_GRAYSCALE)
h,w = img.shape
left = img[:, :w//2]
left_dn = cv2.GaussianBlur(left, (5,5), 0)

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1); plt.imshow(left, cmap='gray'); plt.title('Q9 (a) - Original Image 8a'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(left_dn, cmap='gray'); plt.title('Q9 (a) - Denoised Image 8a (Gaussian Blur)'); plt.axis('off')
plt.show()

#Part (b)
#------------------------------------   
right = img[:, w//2:]
right_dn = cv2.medianBlur(right, 3)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(right, cmap='gray'); plt.title('Q9 (b) - Original Image 8b'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(right_dn, cmap='gray'); plt.title('Q9 (b) - Denoised Image 8b (Median Blur)'); plt.axis('off')
plt.show()

#Part (c) 
#------------------------------------
denoised_img = np.hstack([left_dn, right_dn])
_, th = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(denoised_img, cmap='gray'); plt.title('Q9 (c) - Denoised Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(th, cmap='gray'); plt.title('Q9 (c) - Otsu Thresholding Result'); plt.axis('off')
plt.show()

#Part (d)
#------------------------------------
kernel = np.ones((3,3), np.uint8)
clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(th, cmap='gray'); plt.title('Q9 (d) - Thresholded Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(clean, cmap='gray'); plt.title('Q9 (d) - After Morphology'); plt.axis('off')
plt.show()

#Part (e)
#------------------------------------
num, labels, stats, cents = cv2.connectedComponentsWithStats(clean, connectivity=8)
areas = stats[1:, cv2.CC_STAT_AREA]
count = int(np.sum(areas > 20))

print("Estimated number of rice grains:", count)

plt.figure(figsize=(8, 4)); plt.imshow(clean, cmap="gray"); plt.title("Q9 (e) - Cleaned Mask with Components"); plt.axis("off"); plt.show();

#Display of all
plt.figure(figsize=(8, 4))
plt.subplot(1, 6, 1); plt.imshow(img, cmap='gray'); plt.title('Q9 - Original Image'); plt.axis('off')
plt.subplot(1, 6, 2); plt.imshow(left_dn, cmap='gray'); plt.title('Q9 - Gaussian Blur'); plt.axis('off')
plt.subplot(1, 6, 3); plt.imshow(right_dn, cmap='gray'); plt.title('Q9 - Median Blur'); plt.axis('off')
plt.subplot(1, 6, 4); plt.imshow(denoised_img, cmap='gray'); plt.title('Q9 - Denoised Image'); plt.axis('off')
plt.subplot(1, 6, 5); plt.imshow(th, cmap='gray'); plt.title('Q9 - Otsu Thresholding'); plt.axis('off')
plt.subplot(1, 6, 6); plt.imshow(clean, cmap='gray'); plt.title('Q9 - After Morphology'); plt.axis('off')
plt.show()

#Question 10
#--------------------------------------------------------------------------
#Part (a)
#------------------------------------
img = cv2.imread("data/sapphire.jpg")
if img is None:
    raise FileNotFoundError(f"Could not read: {IMG_PATH}")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([90, 50, 50], dtype=np.uint8)
upper_blue = np.array([140, 255, 255], dtype=np.uint8)
mask_raw = cv2.inRange(hsv, lower_blue, upper_blue)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(1,2,2); plt.title("Initial Blue Mask"); plt.imshow(mask_raw, cmap="gray"); plt.axis("off")
plt.show()

#Part (b)
#------------------------------------
kernel = np.ones((5,5), np.uint8)
mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_open   = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN,  kernel, iterations=1)
mask_filled = np.zeros_like(mask_open)
contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask_filled, contours, -1, color=255, thickness=cv2.FILLED)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("After Close"); plt.imshow(mask_closed, cmap="gray"); plt.axis("off")
plt.subplot(1,3,2); plt.title("After Open"); plt.imshow(mask_open, cmap="gray"); plt.axis("off")
plt.subplot(1,3,3); plt.title("Holes Filled"); plt.imshow(mask_filled, cmap="gray"); plt.axis("off")
plt.show()

#Part (c)
#------------------------------------
num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
areas_px = stats[1:, cv2.CC_STAT_AREA]

print("Number of sapphires detected:", len(areas_px))
print("Areas in pixels:", areas_px)

overlay = cv2.cvtColor(mask_filled, cv2.COLOR_GRAY2BGR)
for i, (cx, cy) in enumerate(centroids[1:], start=1):
    cv2.putText(overlay, f"#{i}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(1,2,2); plt.title("Mask + Labels"); plt.imshow(overlay); plt.axis("off")
plt.show()

#Part (d)
#------------------------------------
f_mm = 8.0    
d_mm = 480.0  
pixel_pitch_mm = 0.005 

scale = (d_mm / f_mm)**2 * (pixel_pitch_mm**2)
areas_mm2 = areas_px * scale

for i, (px_area, mm2_area) in enumerate(zip(areas_px, areas_mm2), start=1):
    print(f"Sapphire {i}: pixels={int(px_area)}, area ≈ {mm2_area:.3f} mm^2 (p={pixel_pitch_mm} mm/pixel)")