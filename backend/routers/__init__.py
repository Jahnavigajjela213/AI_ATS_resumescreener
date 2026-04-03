"""
Backend Routers Package - Exposes all FastAPI routers.
"""
from backend.routers import resume, matching, skills, prediction

__all__ = ["resume", "matching", "skills", "prediction"]
