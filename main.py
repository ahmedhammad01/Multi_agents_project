
import argparse
import logging
import sys
from datetime import datetime
from src.utils.logging_config import setup_logging
from src.utils.config_loader import load_config
from src.agents.coordinator import WorkflowCoordinator

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("🏥 Healthcare Pathway Evaluation Platform Starting...")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Agentic AI Platform for Healthcare Pathways")
    parser.add_argument("--preprocess", action="store_true", help="Run full preprocessing pipeline (stages 1-5)")
    parser.add_argument("--reset", action="store_true", help="Reset data and re-run preprocessing")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        sys.exit(1)

    # Initialize coordinator
    try:
        coordinator = WorkflowCoordinator(config)
        logger.info("✅ All agents initialized successfully")
    except Exception as e:
        logger.error(f"❌ Agent initialization failed: {e}")
        sys.exit(1)

    # Handle reset
    if args.reset:
        logger.info("🧹 Resetting data directories...")
        coordinator.reset_data()

    # Run preprocessing pipeline
    if args.preprocess or args.reset:
        logger.info("📊 Starting data preprocessing pipeline...")
        try:
            result = coordinator.run_preprocessing()
            if result.get("status") == "success":
                logger.info("✅ Preprocessing pipeline completed")
                logger.info(f"📊 Final Quality Score: {result.get('quality_score', 0.0):.3f}")
            else:
                logger.warning("⚠️ Preprocessing completed with issues")
                logger.debug(f"Issues: {result.get('issues', [])}")
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            sys.exit(1)
    else:
        logger.info("🚀 System ready! Launch dashboard with: streamlit run src/dashboard/app.py")

if __name__ == "__main__":
    main()
