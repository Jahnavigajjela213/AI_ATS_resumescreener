"""
run.py - Convenience entry point for the AI Resume Screening API.

Usage:
    python run.py              # Start API server (default: port 8000)
    python run.py --train      # Run training pipeline only
    python run.py --port 9000  # Start on custom port
    python run.py --help       # Show help
"""
import sys
import os
import argparse
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_training():
    """Run the full model training pipeline."""
    print("\n" + "=" * 60)
    print("🚀  AI Resume Screening — Model Training Pipeline")
    print("=" * 60 + "\n")
    from models.train_models import main as train_main
    train_main()


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server."""
    print("\n" + "=" * 60)
    print("🚀  AI Resume Screening — FastAPI Server")
    print(f"    URL  : http://localhost:{port}")
    print(f"    Docs : http://localhost:{port}/docs")
    print(f"    Health: http://localhost:{port}/health")
    print("=" * 60 + "\n")

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main():
    parser = argparse.ArgumentParser(
        description="AI Resume Screening & Skill Gap Prediction System"
    )
    parser.add_argument("--train", action="store_true",
                        help="Run model training pipeline before starting server")
    parser.add_argument("--train-only", action="store_true",
                        help="Run model training pipeline and exit (no server)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind the API server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the API server (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    args = parser.parse_args()

    if args.train_only:
        run_training()
        print("\n✅ Training complete. Exiting.")
        return

    if args.train:
        run_training()

    run_api(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
