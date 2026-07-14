from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from io import BytesIO
from pydantic import BaseModel
import uuid

from src.warp import warp_process
from src.contour import detect_doc_contour
from src.preprocess import *

class ScanRequest(BaseModel):
    image_id: str
    corners: list[list[int]]

image_store = {}

register_heif_opener()

app = FastAPI()

app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )

@app.post("/preview")
async def scan(file: UploadFile = File(...)):
    data = await file.read()
    
    pil = Image.open(BytesIO(data)) 
    img = ImageOps.exif_transpose(pil)
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]

    image_id = str(uuid.uuid4())
    image_store[image_id] = image

    # need to handle this better
    processed_img = preprocess_thresh(image.copy())
    approx, corners, contour = detect_doc_contour(processed_img)

    _, encoded = cv2.imencode(".jpg", image)

    return JSONResponse({
        "image_id": image_id,
        "preview_url": f"/preview/{image_id}",
        "corners": corners.tolist(),
        "width": w,
        "height": h,
    })

@app.get("/preview/{image_id}")
async def get_preview(image_id: str):

    image = image_store.get(image_id)

    if image is None:
        raise HTTPException(404)

    _, encoded = cv2.imencode(".jpg", image)

    return Response(
        encoded.tobytes(),
        media_type="image/jpeg"
    )

@app.post("/scan")
async def scan(request: ScanRequest):
    image = image_store.get(request.image_id)

    if image is None:
        raise HTTPException(
            status_code=404,
            detail="Image expired"
        )
        
    corners = np.array(request.corners)

    result = warp_process(image, corners)

    _, encoded = cv2.imencode(".jpg", result)

    return Response(
        encoded.tobytes(),
        media_type="image/jpeg"
    )