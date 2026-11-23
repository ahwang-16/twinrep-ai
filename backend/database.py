import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, Text, DateTime, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String(120), default="Enterprise Client")
    question = Column(Text, nullable=False)
    retrieved_sources = Column(Text, nullable=True)
    ai_draft = Column(Text, nullable=False)
    final_response = Column(Text, nullable=True)
    status = Column(String(32), default="pending_review")  # pending_review | approved | edited
    created_at = Column(DateTime, default=datetime.utcnow)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def check_db_health() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
