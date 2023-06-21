import pandas as pd
from collections import Counter
from libs.nlp import preprocess_teks, preprocess_file
from services import AppServiceProject
from io import BytesIO
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import io


class CleansingServices(AppServiceProject):
    async def cleansing(self, type, text):
        try:
            if type == "text":
                preprocess = preprocess_teks(text)
                
                data = {
                "data": preprocess
                }
            
                return self.success_response(data)
            else:
                preprocess = preprocess_file(text)

                stream = io.StringIO()
                preprocess.to_csv(stream, index = False)
                response = StreamingResponse(iter([stream.getvalue()]),
                            media_type="text/csv"
                            )
               
                response.headers["Content-Disposition"] = "attachment; filename=data_cleansing.csv"

                return response            
        except Exception as e:
            return self.error_response(e)
