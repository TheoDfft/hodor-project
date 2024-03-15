from sqlalchemy import create_engine, Column, String, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Detection(Base):
    __tablename__ = 'hodor_detections'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    detected_at = Column(DateTime, default=datetime.now)

# Create an SQLite engine
engine = create_engine('sqlite:///hodor_detections.db')

# Create all tables in the engine
Base.metadata.create_all(engine)

# Create a sessionmaker bound to the engine
Session = sessionmaker(bind=engine)

def get_db_session():
    """Create and return a new session."""
    session = Session()
    return session