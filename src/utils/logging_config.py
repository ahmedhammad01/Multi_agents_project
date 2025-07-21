
import logging
import logging.handlers
import json
import os
from typing import Dict, Optional
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Custom formatter for JSON logging"""
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "funcName": record.funcName,
            "lineno": record.lineno
        }
        return json.dumps(log_data)

def setup_logging(config: Optional[Dict] = None):
    """Configure structured JSON logging for the platform"""
    try:
        # Default config if none provided
        config = config or {
            "logging": {
                "level": "INFO",
                "file": "logs/app.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "use_json": True
            }
        }

        # Set up root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, config["logging"]["level"], logging.INFO))

        # Clear any existing handlers
        logger.handlers = []

        # Console handler with standard format
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(config["logging"]["format"])
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler with JSON format
        if config["logging"]["use_json"]:
            os.makedirs(os.path.dirname(config["logging"]["file"]), exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                config["logging"]["file"],
                maxBytes=10_000_000,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(JSONFormatter())
            logger.addHandler(file_handler)

        logger.info("✅ Logging configured successfully")
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        logging.basicConfig(level=logging.INFO)
