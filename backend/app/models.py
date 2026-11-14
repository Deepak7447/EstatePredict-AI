from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func
from .database import Base

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(80), nullable=False)
    predicted_price_usd = Column(Float, nullable=False)
    med_inc = Column(Float, nullable=False)
    house_age = Column(Float, nullable=False)
    ave_rooms = Column(Float, nullable=False)
    ave_bedrms = Column(Float, nullable=False)
    population = Column(Float, nullable=False)
    ave_occup = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
