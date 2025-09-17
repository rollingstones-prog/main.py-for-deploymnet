import json
from pathlib import Path
from supabase import create_client
import os
from dotenv import load_dotenv

# env load
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# path to your local events.jsonl
events_file = Path("events.jsonl")

def migrate_events():
    if not events_file.exists():
        print("❌ events.jsonl file not found")
        return

    with open(events_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        try:
            record = json.loads(line.strip())
            # insert into Supabase
            supabase.table("events").insert(record).execute()
            print(f"✅ Inserted: {record.get('kind')} - {record.get('employee')}")
        except Exception as e:
            print(f"❌ Error inserting record: {e}")

if __name__ == "__main__":
    migrate_events()
