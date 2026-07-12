from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response

import cv2
import numpy as np

from src.scanner import process

from PIL import Image
from pillow_heif import register_heif_opener
from io import BytesIO

register_heif_opener()

app = FastAPI()


@app.get("/")
def root():
    return {"status": "CamScanner server is running"}


@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    data = await file.read()
    
    pil = Image.open(BytesIO(data))
    image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    # print(data)
    # image = cv2.imdecode(
    #     np.frombuffer(data, np.uint8),
    #     cv2.IMREAD_COLOR
    # )

    result = process(image)

    _, encoded = cv2.imencode(".jpg", result)

    return Response(
        encoded.tobytes(),
        media_type="image/jpeg"
    )