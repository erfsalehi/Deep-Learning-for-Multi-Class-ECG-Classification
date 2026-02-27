import requests
import os

url = "https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
# Assuming run from project root (cardiovascular-ai-portfolio)
target_path = "data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv"

# Ensure directory exists
os.makedirs(os.path.dirname(target_path), exist_ok=True)

print(f"Downloading {url} to {target_path}...")
try:
    response = requests.get(url)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(response.content)
        print("Download successful!")
    else:
        print(f"Failed to download. Status code: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
