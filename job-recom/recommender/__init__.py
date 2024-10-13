from .database import init_db
from .recommend import init_recommendation, get_recommendation, INITIALIZED


if not INITIALIZED:
    init_db()
    init_recommendation()
