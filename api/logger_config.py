"""
Level        Purpose
===================================================
DEBUG        Detailed information for debugging
INFO         Key application events (startup, success)
WARNING      Issues that do not interrupt execution
ERROR        Errors that occurred but didn’t stop the program
CRITICAL     Errors that cause the program to crash
"""

import sys

from loguru import logger

# Remove base logger
logger.remove()

# Console output
logger.add(sys.stdout,
           level="INFO",
           format="<green>{time}</green> | <level>{level}</level> | <cyan>{message}</cyan>")

# File output
logger.add("logs/app.log",
           level="DEBUG",
           rotation="1 MB",
           retention="7 days",
           compression="zip",
           format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")

# logger
log = logger
