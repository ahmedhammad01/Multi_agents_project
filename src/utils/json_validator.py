
import logging
from typing import Dict, Any
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

def validate_payload(payload: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate JSON payload against a schema for A2A communication"""
    try:
        validate(instance=payload, schema=schema)
        logger.info("✅ JSON payload validated successfully")
        return True
    except ValidationError as e:
        logger.error(f"❌ JSON payload validation failed: {e.message}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during payload validation: {e}")
        return False
