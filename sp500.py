import pandas as pd
import requests

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {
    "User-Agent": "Mozilla/5.0"
}

html = requests.get(url, headers=headers, timeout=20).text
df = pd.read_html(html)[0]

symbols = df["Symbol"].tolist()
print(symbols)
print(len(symbols))
