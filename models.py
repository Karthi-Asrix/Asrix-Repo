from sqlalchemy import Column, Integer, String
from database import Base

class FastApiData(Base):
    __tablename__ = "fastapi_data"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String)
    context = Column(String)
    response = Column(String)