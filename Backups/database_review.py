from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Detection(Base):
    __tablename__ = 'hodor_detections'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    detected_at = Column(DateTime)

def main():
    # Connect to the database
    engine = create_engine('sqlite:///hodor_detections.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query all detections
    detections = session.query(Detection).all()
    print(f"Number of detections: {len(detections)}")
    # Print the details of each detection
    for detection in detections:
        print(f"ID: {detection.id}, Name: {detection.name}, Detected At: {detection.detected_at}")

if __name__ == "__main__":
    main()
