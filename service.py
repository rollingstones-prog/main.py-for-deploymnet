import os
import time
from config import BOSS_WA_ID, STORAGE_PATHS, WA_TOKEN, WA_PHONE_ID
from utils import load_employees, tts_to_file, whatsapp_send_text, wa_send_audio


def deliver_to_employee(name: str, msg_text: str, override: str = None) -> dict:
    # Load employee data
    emps = load_employees()
    e = emps.get(name)
    
    # Check if the employee exists in the system
    if not e:
        return {"ok": False, "err": f"{name} not in employees.json"}
    
    # Get employee's MSISDN and preferred mode
    msisdn = e["msisdn"]
    mode = (override or e.get("pref") or os.getenv("DELIVERY_DEFAULT","auto")).lower()

    # Voice Mode - Send TTS audio
    if mode == "voice":
        outp = STORAGE_PATHS["media"] / f"tts_{name}_{int(time.time())}.mp3"
        ok = tts_to_file(msg_text, outp)  # Use existing TTS function (text-to-speech)
        
        if ok:
            # Try sending the audio to the employee
            result = wa_send_audio(msisdn, outp, mark_as_voice=True)
            if result["ok"]:
                return {"ok": True, "mode": "voice"}  # Successful audio delivery
            else:
                # Fallback to text if voice sending fails
                whatsapp_send_text(msisdn, msg_text)
                # Notify boss about the fallback
                if BOSS_WA_ID:
                    whatsapp_send_text(BOSS_WA_ID, f"Voice delivery fail {name}. Text bhej diya.")
                return {"ok": True, "mode": "text_fallback"}
        
        # TTS failed, fallback to text
        whatsapp_send_text(msisdn, msg_text)
        return {"ok": True, "mode": "text_fallback"}

    # Text Mode - Directly send the text message
    if mode == "text":
        whatsapp_send_text(msisdn, msg_text)
        return {"ok": True, "mode": "text"}

    # Auto Mode (default to text if no specific preference is set)
    # This is the fallback mode where text is sent automatically
    whatsapp_send_text(msisdn, msg_text)
    return {"ok": True, "mode": "text_auto"}
