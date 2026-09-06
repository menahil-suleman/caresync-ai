"""Quick Supabase connection + pgvector check."""
from dotenv import load_dotenv
load_dotenv()

from config import DBConfig
from sqlalchemy import create_engine, text

print(f"Connecting to: {DBConfig.dsn()[:40]}...")

try:
    engine = create_engine(DBConfig.dsn())
    with engine.connect() as conn:
        # Check Postgres version
        version = conn.execute(text("SELECT version()")).fetchone()[0]
        print(f"✓ Connected: {version[:50]}")

        # Check pgvector
        row = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).fetchone()

        if row:
            print("✓ pgvector: ENABLED")
        else:
            print("✗ pgvector: NOT FOUND")
            print("  → Go to Supabase SQL Editor and run: CREATE EXTENSION IF NOT EXISTS vector;")

except Exception as e:
    print(f"✗ Connection failed: {e}")
