from fastapi import APIRouter
from fastapi import File, UploadFile
from io import StringIO
import pandas as pd
from services.cleansing import CleansingServices

router = APIRouter()

@router.get("/cleansing-text")
async def cleansing_teks_text(
    text: str 
):
    result = await CleansingServices().cleansing(type= "text",text= text)
    return result
    

@router.post("/cleansing-file")
async def cleansing_teks_file(
    file: UploadFile = File(...)
):
    data = pd.read_csv(StringIO(str(file.file.read(), 'latin-1')), encoding='latin-1')
    result = await CleansingServices().cleansing(type= "file",text= data)
    return result