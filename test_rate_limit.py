import requests
import time

url = "https://api.sanaatan.life/ask"
payload = {"question": "What is dharma?"}

print("Testing rate limiting - firing 12 requests...")
for i in range(1, 13):
    r = requests.post(url, json=payload)
    print(f"Request {i}: {r.status_code}")
    if r.status_code == 429:
        print("Rate limit triggered correctly!")
        break
    time.sleep(0.5)

print("Done.")