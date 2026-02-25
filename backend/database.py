from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# ─────────────────────────────────────────────
# STEP 1: Define where database file will be stored
# ─────────────────────────────────────────────

# Get absolute path to backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Database file will be: backend/eam_database.db
DATABASE_PATH = os.path.join(BACKEND_DIR, "eam_database.db")

# Create connection string (tells Python how to connect)
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

print(f"📍 Database will be stored at: {DATABASE_PATH}")

# ─────────────────────────────────────────────
# STEP 2: Create the database engine
# ─────────────────────────────────────────────

# Engine = the connection to the database
# Think of it like opening the filing cabinet
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Required for SQLite
)

# ─────────────────────────────────────────────
# STEP 3: Create session factory
# ─────────────────────────────────────────────

# SessionLocal = creates a "work session" each time we need to access database
# Think of it like: "I need to use the filing cabinet, let me open it"
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ─────────────────────────────────────────────
# STEP 4: Create Base class
# ─────────────────────────────────────────────

# Base = foundation class that all our tables will inherit from
# Think of it like: "All drawers in the filing cabinet follow this design"
Base = declarative_base()

# ─────────────────────────────────────────────
# STEP 5: Helper function to get database session
# ─────────────────────────────────────────────

def get_db():
    """
    Creates a database session and ensures it closes properly
    
    Usage: Every time we want to read/write to database, we use this
    """
    db = SessionLocal()
    try:
        yield db  # Give the session to whoever needs it
    finally:
        db.close()  # Always close when done (important!)