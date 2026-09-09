import cv2
import numpy as np

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


class NoContourFound(Exception):
    pass

class InvalidImage(Exception):
    pass


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

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
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

            aspect_ratio = max(width, height) / float(min(width, height))
            ar_diff = abs(aspect_ratio - target_aspect_ratio)

            # Weight score favoring high area and low aspect ratio deviation
            valid_candidates.append({
                "contour": pts,
                "area": area,
                "ar_diff": ar_diff
            })

    if not valid_candidates:
        raise NoContourFound("Did not find a valid countour")

    # Primary sort: lowest aspect ratio deviation; Secondary: highest area
    valid_candidates.sort(key=lambda x: (x["ar_diff"], -x["area"]))
    
    return valid_candidates[0]["contour"]

def esrgan_upscaling(img):
    """Upscale the image to improve the clarity """
    
    img = cv2.cvtColor(img , cv2.COLOR_GRAY2BGR)
    
    
    model = RRDBNet(num_in_ch = 3, num_out_ch =3 , scale = 4, num_feat = 64 , num_block = 23 , num_grow_ch = 32)
    
    upscaler = RealESRGANer(scale = 4,
                            model_path= "ml_engine/weights/RealESRGAN_x4plus.pth",
                            model = model)
    
    output, _ = upscaler.enhance(img)
    
    
    
    return cv2.cvtColor(output , cv2.COLOR_BGR2GRAY)

def adaptive_binarization(img):
    """Apply adaptive thresholding to a grayscale image."""
    return cv2.adaptiveThreshold(img , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 11 , 2)

def preprocess(img):
    
    if img is None:
        
        raise InvalidImage("Image given is not a valid image")
    
    grey_img = None

    if len(img.shape) == 2:
        grey_img = img
    elif img.shape[2] == 4:
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        grey_img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    
    
    contour =  detect_document_contour(img)
    grey_img = perspective_transform(image = grey_img , contour = contour)
    grey_img = esrgan_upscaling(grey_img)
    grey_img = adaptive_binarization(img = grey_img)
    
    return grey_img

if __name__ == "__main__":
    
    img = cv2.imread("image.png")
    
    final_img = preprocess(img = img)
    
    cv2.imwrite("final_img.png" , final_img)
    
    
    
    



    
    