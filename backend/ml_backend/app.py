from fastapi import FastAPI
from pydantic import BaseModel, field_validator

app = FastAPI()

class ImagePayload(BaseModel):
    image_base64 : str
    doc_id : int

    @field_validator('image_base64')
    def validate_base64(cls , v):
        
        if "data:image" in v:
            v = v.split(',')[1]
    
    
    