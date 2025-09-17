# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

WA_TOKEN = os.getenv("WA_TOKEN")
WA_PHONE_ID = os.getenv("WA_PHONE_ID")
BOSS_WA_ID = os.getenv("BOSS_WA_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

STORAGE_PATHS = {
    "employees": Path("./employees.json"),
    "tasks": Path("./tasks.jsonl"),
    "events": Path("./events.jsonl"),
    "reports": Path("./reports/"),
    "reports_out": Path("./reports/out/"),
    "media": Path("./media/"),
}
