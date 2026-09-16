import pytesseract
import cv2

config_mrz = config_mrz = (
    '--tessdata-dir "ml_engine/tessdata" '
    "-l mrz --oem 1 --psm 6 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< "
    "-c preserve_interword_spaces=0"
)

def extractMRZ(image) -> str:
    
    x = pytesseract.image_to_string(image , config = config_mrz)
    
    print(x)
    
    return x

def extractImage(image_path : str) -> None:
    pass
    
    
    
    
    
    
    
    