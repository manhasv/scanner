import numpy as np
import cv2

__all__ = ['gray_world_white_balance', 'homomorphic_filter', 'contrast_enhance', 
'morphological_illumination', 'illumination_correction']


def gray_world_white_balance(img):
    # 1. Convert to float to prevent uint8 overflow issues during math operations
    img_float = img.astype(np.float32)
    
    # 2. Calculate the average value for each color channel (Blue, Green, Red)
    # axis=(0,1) averages across rows and columns, leaving a 1D array of 3 channels
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])
    
    # 3. Calculate the overall baseline gray target (average of all 3 channel averages)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    # 4. Compute scaling factors for each individual channel
    # Safeguard against zero-division in completely black channels
    scale_b = avg_gray / (avg_b if avg_b != 0 else 1.0)
    scale_g = avg_gray / (avg_g if avg_g != 0 else 1.0)
    scale_r = avg_gray / (avg_r if avg_r != 0 else 1.0)
    
    # 5. Apply scaling factors to equalize channel weights
    img_float[:, :, 0] *= scale_b
    img_float[:, :, 1] *= scale_g
    img_float[:, :, 2] *= scale_r
    
    # 6. Clip values to [0, 255] range and cast back to unsigned 8-bit integer
    corrected_img = np.clip(img_float, 0, 255).astype(np.uint8)
    
    return corrected_img

def contrast_enhance(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced_img

def homomorphic_filter(img, alpha=0.75, beta=1.25, cutoff=80, order=2):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    # 1. Log Transform: log(I(x,y)) = log(i(x,y)) + log(r(x,y))
    img_log = np.log(L + 1)
    
    # 2. FFT to frequency domain
    fft = np.fft.fftshift(np.fft.fft2(img_log))
    
    # 3. Create Butterworth High-Pass Filter mask
    rows, cols = L.shape
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2

    y, x = np.meshgrid(u, v, indexing="ij")
    D = np.sqrt(x**2 + y**2)
    H = 1 - 1/(1 + (D/cutoff)**(2*order))
    H = alpha + (beta - alpha) * H
    # 4. Apply filter, Inverse FFT, and exponentiate
    filtered_fft = fft * H
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))
    img_exp = np.exp(img_back) - 1.0

    # 5. Normalize result
    L_filtered = cv2.normalize(
        np.clip(img_exp, 0, None),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    lab[:, :, 0] = L_filtered

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def morphological_illumination(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    background = cv2.morphologyEx(
        gray,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (151, 151))
    )

    corrected = cv2.divide(gray, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    final_img = clahe.apply(corrected)
    return final_img

def illumination_correction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=75)

    gray_f = gray.astype(np.float32)
    bg_f = background.astype(np.float32)

    corrected = gray_f / (bg_f + 1)
    corrected *= np.mean(bg_f)

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(16,16)
    )

    result = clahe.apply(corrected)

    return result

def local_contrast_normalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = gray.astype(np.float32)

    sigma = 25

    mean = cv2.GaussianBlur(gray, (0,0), sigma)

    mean_sq = cv2.GaussianBlur(gray**2, (0,0), sigma)

    variance = mean_sq - mean**2

    variance = np.maximum(variance, 1e-6)

    std = np.sqrt(variance)

    normalized = (gray - mean) / (std + 1)

    normalized -= normalized.min()

    normalized /= normalized.max()

    normalized *= 255

    return normalized.astype(np.uint8)