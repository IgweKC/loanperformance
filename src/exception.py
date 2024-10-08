# Catch all exception in a single style (for simplicity)
import sys

def error_message_detail(error, error_detail:sys):
    '''
    Catch exception in our context showing file, line number and msg.
    '''
    _,_,exc_tb=error_detail.exc_info()

    # get the error-infested  filename from Traceback object
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error in python script: [{0}], line number: [{1}], error message: [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message  
