from pathlib import Path
import cv2
import pytesseract

# 1. Load image
img = cv2.imread("MRZ.jpg", cv2.IMREAD_GRAYSCALE)

# 2. Add white padding around the text
padded = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)

# 3. Smooth jagged interpolation artifacts
blurred = cv2.GaussianBlur(padded, (3, 3), 0)
_, cleaned = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Configurations to test
tessdata_dir = "/home/ashishd/SIH/Robust-OCR-For-Fake-Identity-and-Document-Screening/ml_engine/tessdata"

# Test A: Custom MRZ model (Forced LSTM, single text block)
config_mrz = (
    f'--tessdata-dir "{tessdata_dir}" '
    "-l mrz --oem 1 --psm 6 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< "
    "-c preserve_interword_spaces=0"
)

# Test B: Standard English model (often more resilient to font distortions)
config_eng = (
    "-l eng --oem 1 --psm 6 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< "
    "-c preserve_interword_spaces=0"
)

print("--- Testing MRZ Model ---")
print(pytesseract.image_to_string(cleaned, config=config_mrz))

print("--- Testing ENG Fallback ---")
print(pytesseract.image_to_string(cleaned, config=config_eng))