
import logging
import json
from typing import Dict
import warnings

# Suppress AIF360 optional dependency warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No module named 'tensorflow'")
warnings.filterwarnings("ignore", category=UserWarning, message="No module named 'inFairness'")
warnings.filterwarnings("ignore", category=UserWarning, message="No module named 'fairlearn'")

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": record.created,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno
        }
        return json.dumps(log_record)

from typing import Optional

def setup_logging(config: Optional[Dict] = None):
    """Configure JSON logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(config.get("log_file", "logs/app.log") if config else "logs/app.log")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    
    logger.info("✅ Logging configured successfully")
