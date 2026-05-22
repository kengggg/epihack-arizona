from datetime import date
import json
from layer_2.story_1 import generate_message

user_report = {
    "fever": "YES",
    "cough_congestion": "YES",
    "rash": "NO",          # ignored (not YES)
    "loss_of_smell_or_taste": "YES",
}
user_profile = {"age": 34, "sex": "F"}

payloads, message = generate_message(
    db_path="data/epihack.db",
    user_report=user_report,
    user_profile=user_profile,
    area="850",                       # postal prefix — "850" covers many AZ ZIPs
    anchor_date=date(2026, 5, 21),    # the fixture's last date
)

print("=== JSON PAYLOADS (one per YES symptom) ===")
print(json.dumps(payloads, indent=2))
print("\n=== LLM MESSAGE (the user-facing advisory) ===")
print(message)