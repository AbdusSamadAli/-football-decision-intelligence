import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing FOOTBALL_DATA_API_KEY in .env")

url = "https://api.football-data.org/v4/competitions/PL"

response = requests.get(
    url,
    headers={"X-Auth-Token": API_KEY},
    timeout=15
)

print("HTTP Status", response.status_code)
response.raise_for_status()

data = response.json()

print("Competition:", data["name"])
print("Country:", data["area"]["name"])
print("Competition code:", data["code"])
