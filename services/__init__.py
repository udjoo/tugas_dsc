from utils.responses import (
    success_handler,
    error_handler,
    bad_request_handler,
    unauthorized_handler,
)

class AppServiceProject():
    def error_response(self, e):
        return error_handler(e)

    def bad_request_response(self, **kwargs):
        return bad_request_handler(kwargs)

    def success_response(self, data, **kwargs):
        return success_handler(data, kwargs)

    def unauthorized_response(self, **kwargs):
        return unauthorized_handler(kwargs)
