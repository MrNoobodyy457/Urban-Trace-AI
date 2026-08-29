"""
DATABASE SETUP - PLACE IN: src/database.py
SQLAlchemy models for UrbanTrace AI
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

# Database path (relative to project root)
db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'nabrad.db')
DATABASE_URL = f"sqlite:///{db_path}"

# Create engine
engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()

# Create session factory
SessionLocal = sessionmaker(bind=engine)


class Vehicle(Base):
    """Global vehicle record across all cameras"""
    __tablename__ = 'vehicles'
    
    id = Column(Integer, primary_key=True)
    global_id = Column(String, unique=True, nullable=False)  # e.g., "MH02AB1234"
    first_seen = Column(DateTime, default=datetime.now)
    last_seen = Column(DateTime, default=datetime.now)
    total_observations = Column(Integer, default=0)


class Observation(Base):
    """Single vehicle sighting in a camera"""
    __tablename__ = 'observations'
    
    id = Column(Integer, primary_key=True)
    global_vehicle_id = Column(String, nullable=False)  # Link to global_id
    camera_id = Column(String, nullable=False)  # e.g., "cam_north_1"
    timestamp = Column(DateTime, default=datetime.now)
    plate = Column(String)  # Detected license plate
    plate_confidence = Column(Float)  # OCR confidence (0-1)
    direction = Column(String)  # e.g., "north", "south"
    bbox_x1 = Column(Integer)  # Bounding box for visualization
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)


class Trajectory(Base):
    """Complete journey of a vehicle"""
    __tablename__ = 'trajectories'
    
    id = Column(Integer, primary_key=True)
    global_vehicle_id = Column(String, nullable=False)
    path = Column(String)  # Comma-separated camera IDs: "cam_A,cam_C,cam_F"
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    total_time_seconds = Column(Float)


# Create all tables
def init_db():
    """Initialize database - run this once"""
    Base.metadata.create_all(engine)
    print("✅ Database initialized!")


# Test connection
if __name__ == "__main__":
    init_db()
    
    # Test session
    session = SessionLocal()
    print(f"✅ Session created: {session}")
    session.close()
