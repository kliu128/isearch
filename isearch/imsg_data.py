# SQLAlchemy tables for iMessage

from sqlalchemy import Column, BLOB, String, Integer

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MessageEmbedding(Base):
    __tablename__ = 'message_embedding'

    guid = Column(String, nullable=False)
    embed = Column(BLOB, nullable=False)
    model_ver = Column(Integer, nullable=False)
