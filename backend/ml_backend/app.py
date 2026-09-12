from fastapi import FastAPI , Header , HTTPException
from pydantic import BaseModel, field_validator
from typing import Union
import base64
import cv2
import numpy as np
from ml_engine.module_1.preprocessor import preprocess
from ml_engine.module_1 import extractor
from ml_engine.module_2 import validator


app = FastAPI()

class ImagePayload(BaseModel):
    image_base64 : str


    @field_validator('image_base64')
    def validate_base64(cls , v):
        
        if "data:image" in v:
            v = v.split(',')[1]
        return v
        
class Response(BaseModel):
    status : int
    validity : bool

class PassPortResponse(Response):
    surname : str
    name : str


            
@app.post("/predict" , response_model = Union[PassPortResponse , Response])
def predict_(payload : ImagePayload ,
             doc_id : int = Header(...),
             request_id : int = Header(...)):
    
    img_bytes = base64.b64decode(payload.image_base64)
    img_arr = np.frombuffer(img_bytes , dtype= np.uint8)
    
    img = cv2.imdecode(img_arr , flags = cv2.IMREAD_COLOR)
    
    if doc_id not in [1,2,3]: 
            raise HTTPException(status_code = 400)
    
    if doc_id == 3:
        print("[+] Starting preprocessing ... ")
        
        final_img, mrz = preprocess(img = img , r_id=request_id)
        
        MRZ = extractor.extractMRZ(mrz)
        
        status = validator.validate_mrz(mrz = MRZ)
        
        if status["status"]: # type: ignore
            return PassPortResponse(
                status = 200,
                validity = status["status"],  # type: ignore
                surname = status["surname"], # type: ignore
                name = status["name"] # type: ignore
            )
            
        return Response(status = 400,
                        validity = False)
        
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)  
        
    
    
        
        
        
        
        
         