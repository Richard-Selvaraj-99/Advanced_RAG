import logging
import sys
def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        # 1. This prints to your screen
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. This creates the .log file
        file_handler = logging.FileHandler("project_history.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
# This part "calls" the function you wrote
log = get_logger(__name__)

# This actually sends a message to the VS Code Terminal
log.info("Hello! This is a test log message.")
log.warning("Something might be wrong!")