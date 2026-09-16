import cv2
import numpy as np

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

import pytesseract

import os

os.environ["TESSDATA_PREFIX"] = "ml_engine/tessdata"

class NoContourFound(Exception):
    pass

class InvalidImage(Exception):
    pass

class unknownShape(Exception):
    pass

config_mrz = (
    '--tessdata-dir "ml_engine/tessdata" '
    "-l mrz --oem 1 --psm 6 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< "
    "-c preserve_interword_spaces=0"
)

def order_points(pts):
    """
    Orders 4 points in the exact order: top-left, top-right, bottom-right, bottom-left.
    Expects an array of shape (4, 2).
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left point has the smallest sum, bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right point has the smallest difference, bottom-left has the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def perspective_transform(image, contour):
    """
    Applies a top-down birds-eye view transformation given a 4-point contour.
    """
    # Reshape the contour to a (4, 2) array of coordinates
    pts = contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for the top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and warp the image
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def correct_orientation(img) :

    if len(img.shape) < 2:
        raise unknownShape("img array has less than 2 dimensions")
    
    h , w = img.shape[:2]
    
    if w > h:
        print("[+] Image is horizontal, rotating it 90 degrees clockwise")
        return correct_orientation(cv2.rotate(img , cv2.ROTATE_90_CLOCKWISE))
    
    top_strip = img[: int(h*0.20), :]
    bot_strip = img[int(h*0.80) : , :]
    
    top_energy = np.mean(np.abs(cv2.Sobel(top_strip , cv2.CV_32F , 1 , 0 , ksize=3)))
    bot_energy = np.mean(np.abs(cv2.Sobel(bot_strip , cv2.CV_32F, 1 , 0 , ksize =3)))
    
    if top_energy > bot_energy:
        print("[+] Image is upside down rotating it one 180 degree ")
        
        return cv2.rotate(img , cv2.ROTATE_180)
    
    return img
        

def detect_document_contour(
    img: np.ndarray, 
    target_aspect_ratio: float = 1.475, 
    min_area_ratio: float = 0.10
) -> np.ndarray:
    """
    Detects and returns the single best-matching 4-point document contour from an in-memory image.
    
    Contours are prioritized by bounding area and matched against the target aspect ratio
    (defaulting to ISO/IEC 7810 ID-1 ~ 1.585).
    
    Returns:
        np.ndarray of shape (4, 2) containing ordered corner points, or None if not found.
    """
    
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Input 'img' must be a valid numpy array.")
    

    h_img, w_img = img.shape[:2]
    total_area = h_img * w_img
    min_area = total_area * min_area_ratio

    # Handle grayscale or multi-channel arrays
    if len(img.shape) == 2:
        gray = img
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order and filter out minor background noise
    valid_candidates = []
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in sorted_contours:
        area = cv2.contourArea(c)
        if area < min_area:
            break  # Subsequent contours will be even smaller

        peri = cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, closed=True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            rect = cv2.minAreaRect(pts)
            (_, _), (width, height), _ = rect
            
            if width == 0 or height == 0:
                continue

            aspect_ratio = max(width,height)/min(height,width)
            ar_diff = abs(aspect_ratio - target_aspect_ratio)
            # Weight score favoring bottom most , high area and low aspect ratio deviation
            valid_candidates.append({
                "contour": pts,
                "center_y" : np.mean(pts[: ,1]),
                "area": area,
                "ar_diff": ar_diff
            })

    if not valid_candidates:
        raise NoContourFound("Did not find a valid countour")

    # Primary sort: lowest aspect ratio deviation; Secondary: highest area
    valid_candidates.sort(key=lambda x: (x["ar_diff"], -x["center_y"],-x["area"]))
    
    return valid_candidates[0]["contour"]

def esrgan_upscaling(img):
    """Upscale the image to improve the clarity """
    
    img = cv2.cvtColor(img , cv2.COLOR_GRAY2BGR)
    
    
    model = RRDBNet(num_in_ch = 3, num_out_ch =3 , scale = 4, num_feat = 64 , num_block = 23 , num_grow_ch = 32)
    
    upscaler = RealESRGANer(scale = 4,
                            model_path= "ml_engine/weights/RealESRGAN_x4plus.pth",
                            tile = 256,
                            model = model)
    
    output, _ = upscaler.enhance(img)
    
    
    
    return cv2.cvtColor(output , cv2.COLOR_BGR2GRAY)

def preprocess_mrz(img_path: str) -> str:
    # Load grayscale directly
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 1. DO NOT apply hard threshold/Otsu. Tesseract's LSTM engine needs grayscale antialiasing.
    # 2. Add clean white padding so edge characters are not clipped
    padded = cv2.copyMakeBorder(
        img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255  # type: ignore
    ) 

    # 3. Light normalization to maximize contrast without destroying gradients
    norm = cv2.normalize(padded, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  # type: ignore

    cv2.imwrite(img_path, norm)
    return img_path


def mrz_roi(img: np.ndarray, r_id: int):
    """Extracts the bottom ~16% of the opened passport document,

    skipping the outer borders to isolate the text lines.
    """
    h_img, w_img = img.shape[:2]

    # For an opened passport (1476x2079), the MRZ zone starts at ~84% height
    # We inset by 60px horizontally and vertically to clear any bounding box borders
    y_start = int(h_img * 0.84)
    y_end = int(h_img * 0.98)
    x_start = int(w_img * 0.04)
    x_end = int(w_img * 0.96)

    return img[y_start:y_end, x_start:x_end]

    
def preprocess(img, r_id : int):
    
    if img is None:
        
        raise InvalidImage("Image given is not a valid image")
    
    grey_img = None

    if len(img.shape) == 2:
        grey_img = img
    elif img.shape[2] == 4:
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        grey_img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    print("[+] Detecting Contour")
    contour =  detect_document_contour(img)
    print("[+] Performing prespective transform")
    grey_img = perspective_transform(image = grey_img , contour = contour)
    
    print("[+] Correcting Orientation")
            
    grey_img = correct_orientation(grey_img)
    
    
    # print("[+] Upscaling")
    # grey_img = esrgan_upscaling(grey_img)
    
    print(f"[+] Saving MRZ to Images/MRZ{r_id}.jpg")
    
    mrz = mrz_roi(grey_img , r_id)
    
    print("[+] Completed")
    
    return grey_img , mrz

if __name__ == "__main__":
    
    img = cv2.imread("image.png")
    
    final_img , _ = preprocess(img = img, r_id = 0)
    
    cv2.imwrite("final_img.png" , final_img)
    
    
    
    



    
    