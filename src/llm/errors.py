class LLMError(Exception):
    #Exceçao base para erros relacionados a LLM
    def __init__(self, message: str, error_type: str = "unknown", attempts: int = 1):
        super().__init__(message) 
        self.error_type = error_type;self.attempts = attempts

class LLMTimeoutError(LLMError): pass
class LLMRateLimitError(LLMError): pass
class LLMServerError(LLMError): pass
class LLMBadRequestError(LLMError): pass
class LLMFormatError(LLMError): pass
class ParseError(LLMError): pass
class LLMRetryError(LLMError):pass