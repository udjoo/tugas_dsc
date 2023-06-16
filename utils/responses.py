from fastapi import status
from fastapi.responses import JSONResponse


def success_handler(data, kwargs):
    content = {
        "ok": True,
        "code": status.HTTP_200_OK,
        "data": data,
        "message": "Success",
    }
    content.update(kwargs)
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


def bad_request_handler(kwargs):
    content = {
        "ok": False,
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Bad Request",
    }
    content.update(kwargs)
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)


def unauthorized_handler(kwargs):
    content = {
        "ok": False,
        "code": status.HTTP_401_UNAUTHORIZED,
        "message": "Email or password is wrong",
    }
    content.update(kwargs)
    return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content=content)


def error_handler(exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "ok": False,
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "Internal Server Error",
        },
    )
