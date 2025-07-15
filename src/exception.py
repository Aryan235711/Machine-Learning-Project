import sys
from src.logger import logging # You're importing the module, not a specific logger instance


# It's better to get the configured logger instance
# rather than relying on the root logger implicitly or trying to reconfigure it.
# If you don't explicitly retrieve 'app_logger' here, you'll be using the root logger
# which was configured in logger.py when it was first imported.
# For simplicity, we'll continue using the implicitly configured root logger as your setup does.

def error_message_detail(error, error_detail:sys): # error_detail should be the tuple from sys.exc_info()
    _, _, exc_tb = error_detail.exc_info() # Correctly unpack the tuple
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    # You can also include the module name for better context
    module_name = exc_tb.tb_frame.f_globals['__name__'] 
    
    error_message = (
        f"Error occurred in module [{module_name}] "
        f"script name [{file_name}] "
        f"at line number [{line_number}] "
        f"error message: [{str(error)}]"
    )
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        # error_detail here should be the tuple from sys.exc_info()
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    


# This is just an example to show how you might use the CustomException class.
if __name__ == '__main__':
    try:
        a = 1/0
    except Exception as e:
        # Correct way to log your custom exception:
        # 1. Raise CustomException
        # 2. Catch CustomException
        # 3. Use logging.error() with exc_info=True for the traceback of CustomException
        #    AND the original ZeroDivisionError (due to 'from e')
        try:
            # Raise the custom exception, passing the original exception and sys.exc_info()
            raise CustomException(f"Mathematical operation error: {e}", error_detail=sys.exc_info()) from e
        except CustomException as ce:
            # Now log the CustomException object (whose __str__ contains the formatted message)
            # and request the full traceback with exc_info=True.
            logging.error(str(ce), exc_info=True)
            # Alternatively, if CustomException's __str__ provides the formatted message AND the traceback:
            # logging.error(str(ce)) # But this won't show the chained exception traceback automatically
            
            # The most robust way to log the CustomException AND its original cause:
            # logging.error(f"Custom exception caught: {ce}", exc_info=True)
            # or if you want just the CustomException's internal formatted message as the primary log message
            # logging.error(str(ce), exc_info=True) 

