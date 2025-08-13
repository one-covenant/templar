import asyncio
from botocore import exceptions as boto_excepts

import tplr


def handle_s3_exceptions(
    e: Exception,
    func_name: str,
) -> bool:
    """A centralized helper function to log and handle exceptions."""
    purge = False

    base_message = f"Function '{func_name} failed:"
    if isinstance(e, boto_excepts.ClientError):
        error_code = e.response.get("Error", {}).get("Code")
        error_message = e.response.get("Error", {}).get("Message")
        tplr.logger.exception(f"{base_message} {error_message} (Code: {error_code})")
        purge = True
    elif isinstance(e, boto_excepts.ConnectionClosedError):
        tplr.logger.exception(f"{base_message} Connection Closed Error: {e}")
        purge = True
    elif isinstance(e, boto_excepts.NoCredentialsError):
        tplr.logger.exception(
            f"{base_message} No AWS credentials found. Please configure your credentials."
        )
    elif isinstance(e, boto_excepts.ParamValidationError):
        tplr.logger.exception(f"{base_message} Parameter Validation Error: {e}")
    elif isinstance(e, asyncio.TimeoutError):
        tplr.logger.exception(f"{base_message} ")
    else:
        tplr.logger.exception(f"{base_message} An unexpected error occurred: {e}")
    return purge

def handle_general_exceptions(e: Exception, func_name: str) -> None:
    """A centralized helper function to log and handle common Python exceptions."""
    base_message = f"Function '{func_name}' encountered an error:"

    if isinstance(e, IndexError):
        tplr.logger.exception(f"{base_message} IndexError - likely access to a list/tuple with an out-of-bounds index: {e}")
    elif isinstance(e, ValueError):
        tplr.logger.exception(f"{base_message} ValueError - an argument is of the correct type but has an inappropriate value: {e}")
    elif isinstance(e, KeyError):
        tplr.logger.exception(f"{base_message} KeyError - likely access to a dictionary with a non-existent key: {e}")
    elif isinstance(e, TypeError):
        tplr.logger.exception(f"{base_message} TypeError - an operation or function is applied to an object of inappropriate type: {e}")
    elif isinstance(e, AttributeError):
        tplr.logger.exception(f"{base_message} AttributeError - an attribute reference or assignment fails: {e}")
    elif isinstance(e, FileNotFoundError):
        tplr.logger.exception(f"{base_message} FileNotFoundError - a file or directory is requested but doesn't exist: {e}")
    else:
        tplr.logger.exception(f"{base_message} An unexpected error occurred: {e}")