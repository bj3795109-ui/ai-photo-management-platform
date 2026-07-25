"""
One-time migration: adds the created_at column to the photos table.

Run this once after updating the code:
    python scripts/add_created_at.py

It's idempotent — safe to run multiple times.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db.database import engine

DDL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'photos' AND column_name = 'created_at'
    ) THEN
        ALTER TABLE photos ADD COLUMN created_at TIMESTAMP NOT NULL DEFAULT now();
    END IF;
END $$;
"""

if __name__ == "__main__":
    with engine.connect() as conn:
        conn.execute(DDL)
        conn.commit()
    print("✓ created_at column ensured on photos table")
