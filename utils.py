from datetime import datetime, timezone
import os
from pathlib import Path

import supabase

from dotenv import load_dotenv
load_dotenv(override=True)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
WA_TOKEN = os.getenv("WA_TOKEN")
WA_PHONE_ID = os.getenv("WA_PHONE_ID")

def now_iso():
    return datetime.now(timezone.utc).isoformat()

import requests
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

STORAGE_PATHS = {
    "employees": Path("./employees.json"),
    "tasks": Path("./tasks.jsonl"),
    "events": Path("./events.jsonl"),
    "reports": Path("./reports/"),
    "reports_out": Path("./reports/out/"),
    "media": Path("./media/"),
}


def wa_send_audio(to_msisdn: str, media_id: str | None = None, link: str | None = None, mark_as_voice: bool = False) -> dict:
    """
    Send an audio file via WhatsApp Cloud API.
    - Either pass media_id (uploaded with wa_upload_media) OR link (public URL).
    - If mark_as_voice=True, WhatsApp treats it as a voice note.
    Returns dict: {"ok": True, "result": {...}} or {"ok": False, "error": "..."}
    """
    url = f"https://graph.facebook.com/v23.0/{WA_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to_msisdn,
        "type": "audio",
        "audio": {}
    }

    if media_id:
        payload["audio"]["id"] = media_id
    elif link:
        payload["audio"]["link"] = link
    else:
        return {"ok": False, "error": "No media_id or link provided"}

    if mark_as_voice:
        payload["audio"]["voice"] = True

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        if "messages" in result:
            return {"ok": True, "result": result}
        else:
            return {"ok": False, "error": result}

    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Request error: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Unexpected error: {e}"}

def mask_msisdn(msisdn: str) -> str:
    if len(msisdn) >= 7:
        return msisdn[:3] + "***" + msisdn[-4:]
    return msisdn

def parse_name_list(names_str: str) -> list:
    if not names_str.strip():
        return []
    return [name.strip() for name in names_str.split(",") if name.strip()]

def handle_whatsapp_error(error_info, to_msisdn: str):
    # error_info kabhi dict, kabhi raw string ho sakta hai — normalize:
    if isinstance(error_info, str):
        try:
            error_info = json.loads(error_info)
        except Exception:
            error_info = {"error": {"message": error_info}}

    err = error_info.get("error", {}) or {}
    code = err.get("code")
    sub  = err.get("error_subcode")
    msg  = err.get("message") or "Unknown error"

    append_jsonl(STORAGE_PATHS["events"], {
        "at": now_iso(), "kind": "WA_ERR",
        "code": code, "subcode": sub, "message": msg
    })

    # SPECIFIC HANDLING:
    if code == 190:
        # Token expired/invalid
        # NOTE: Yahan user ko dobara whatsapp_send_text na call karo; sirf boss ko later inform karna ho to safe jagah pe karo
        print("WA ERROR 190: Token invalid/expired. Refresh WA_TOKEN.")
        return

    if code == 131047:
        # 24h session window band
        print("WA ERROR 131047: 24h session khatam. Template chahiye.")
        return

    if code == 10 or code == 200 or code == 294:  # generic permission/limit
        print(f"WA PERMISSION/LIMIT ERROR: {msg}")
        return

    # default
    print(f"WA ERROR: {code} / {sub} → {msg}")

def append_event(record: dict):
    print("👉 append_event called with:", record)   # Debug
    try:
        resp = supabase.table("events").insert(record).execute()
        print("👉 Supabase response:", resp)
    except Exception as e:
        print(f"❌ Supabase insert error: {e}")

def whatsapp_send_text(to_msisdn: str, text: str , emp_name: str = None):
    token = os.getenv("WA_TOKEN") or ""
    phone_id = os.getenv("WA_PHONE_ID") or ""
    url = f"https://graph.facebook.com/v23.0/{phone_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_msisdn,
        "type": "text",
        "text": {"body": text}
    }
    headers = {"Authorization": f"Bearer {token}"}
    # debug mask
    if token:
        print("WA_TOKEN(use):", token[:8] + "..." + token[-6:])
    else:
        print("WA_TOKEN(use): NONE")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        try:
            data = resp.json()
        except Exception:
            data = {"status_code": resp.status_code, "text": resp.text}

        
        employees = load_employees()
        if not emp_name:
            for name, info in employees.items():
                if info.get("msisdn") == to_msisdn:
                    emp_name = name
                    break

        record = {
            "at": now_iso(),
            "kind": "WA_SEND",
            "employee": emp_name,
            "msisdn": to_msisdn,
            "to": to_msisdn,
            "payload": {"text": {"body": text}, "type": "text"}
        }

        

        append_event(record)
        if resp.status_code >= 400:
            handle_whatsapp_error(data, to_msisdn)  # ALWAYS dict-like
        return data

    except requests.RequestException as e:
        # network-level error
        data = {"error": {"message": str(e), "code": -1}}
        append_jsonl(STORAGE_PATHS["events"], {
            "at": now_iso(), "kind": "WA_SEND_ERR", "to": to_msisdn, "err": str(e)
        })
        # yahan koi user-msg dubara mat bhejna
        return data


def tts_to_mp3(text, out_path, voice_id=ELEVENLABS_VOICE_ID):
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}, "output_format": "mp3"}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return str(out_path)
        return None
    except Exception as e:
        print("TTS error:", e)
        return None


def tts_to_file(text, out_path):
    try:
        result = tts_to_mp3(text, out_path)
        if result and out_path.exists() and out_path.stat().st_size < 16 * 1024 * 1024:
            return out_path
        else:
            print(f"Warning: TTS file too large or not created: {out_path}")
            return None
    except Exception as e:
        append_jsonl(STORAGE_PATHS["events"], {"error": "AI_ERR", "details": str(e)})
        return None

import json
import time
from pathlib import Path

def append_jsonl(path, data):
    data["timestamp"] = now_iso()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
def load_employees():
    try:
        with open(STORAGE_PATHS["employees"], "r", encoding="utf-8") as f:
            raw = json.load(f)
            upgraded = {}
            default_pref = os.getenv("DELIVERY_DEFAULT", "auto")
            for name, val in raw.items():
                if isinstance(val, str):
                    upgraded[name] = {"msisdn": val, "pref": default_pref}
                else:
                    val.setdefault("pref", default_pref)
                    upgraded[name] = val
            return upgraded
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def debounce_message(msg_id, seen_set, seen_path):
    if msg_id in seen_set:
        return True
    seen_set.add(msg_id)
    with open(seen_path, "w", encoding="utf-8") as f:
        json.dump(list(seen_set), f, ensure_ascii=False)
    return False

def get_employee_name_by_msisdn(msisdn, employees):
    """Return employee name for a given msisdn, or None if not found."""
    for name, info in employees.items():
        if info.get("msisdn") == msisdn:
            return name
    return None


