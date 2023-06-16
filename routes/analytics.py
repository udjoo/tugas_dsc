from fastapi import APIRouter
from fastapi.responses import JSONResponse 
from services.analytics import AnalyticServices
from fastapi import File, UploadFile
import pandas as pd
from io import StringIO

router = APIRouter()

@router.get("/sentiment-mlp")
async def sentiment_analytics_mlp(
    text: str
):
    result = await AnalyticServices().get_sentiment_analytics(text= text, type= 'mlp')
    return result

@router.post("/sentiment-mlp-file")
async def sentiment_analytics_mlp_file(
    file: UploadFile = File(...)
):
    data = pd.read_csv(StringIO(str(file.file.read(), 'latin-1')), encoding='latin-1')
    result = await AnalyticServices().get_sentiment_analytics_file(input= data,type= "mlp")
    return result

@router.get("/sentiment-rnn")
async def sentiment_analytics_rnn(
    text: str
):
    result = await AnalyticServices().get_sentiment_analytics(text= text, type= 'rnn')
    return result

@router.post("/sentiment-rnn-file")
async def sentiment_analytics_rnn_file(
    file: UploadFile = File(...)
):
    data = pd.read_csv(StringIO(str(file.file.read(), 'latin-1')), encoding='latin-1')
    result = await AnalyticServices().get_sentiment_analytics_file(input= data,type= "rnn")
    return result

@router.get("/sentiment-lstm")
async def sentiment_analytics_lstm(
    text: str
):
    result = await AnalyticServices().get_sentiment_analytics(text= text, type= 'lstm')
    return result


@router.post("/sentiment-lstm-file")
async def sentiment_analytics_lstm_file(
    file: UploadFile = File(...)
):
    data = pd.read_csv(StringIO(str(file.file.read(), 'latin-1')), encoding='latin-1')
    result = await AnalyticServices().get_sentiment_analytics_file(input= data,type= "lstm")
    return result