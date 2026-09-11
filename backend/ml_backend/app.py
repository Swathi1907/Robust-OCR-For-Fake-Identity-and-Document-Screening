from fastapi import FastAPI , Header , HTTPException
from pydantic import BaseModel, field_validator
import base64
import cv2
import numpy as np

app = FastAPI()

class ImagePayload(BaseModel):
    image_base64 : str


    @field_validator('image_base64')
    def validate_base64(cls , v):
        
        if "data:image" in v:
            v = v.split(',')[1]
            
@app.get("/predict")
def predict_(payload : ImagePayload ,
             doc_id : int = Header(...)):
    
    img_bytes = base64.b64decode(payload.image_base64)
    img_arr = np.frombuffer(img_bytes , dtype= np.uint8)
    
    img = cv2.imdecode(img_arr , flags = cv2.IMREAD_COLOR)
    
    
    
    if doc_id == 3:
         