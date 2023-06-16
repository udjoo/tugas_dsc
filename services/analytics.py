import pandas as pd
from collections import Counter
from services import AppServiceProject
from models.sentiment import get_sentiment, get_sentiment_file

class AnalyticServices(AppServiceProject):
    async def get_sentiment_analytics(self, text, type):
        try:
            sentiment = await get_sentiment(text, type)
            
            data = {
                "data": sentiment
            }
            
            return self.success_response(data)
        except Exception as e:
            return self.error_response(e)

    async def get_sentiment_analytics_file(self, input, type):
        try:
            original_text, sentiment = await get_sentiment_file(input, type)

            data = {
                "original_text": original_text,
                "sentiment": sentiment
            }

            return self.success_response(data)
        except Exception as e:
            return self.error_response(e)