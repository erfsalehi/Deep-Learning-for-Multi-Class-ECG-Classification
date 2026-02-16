import wfdb
import os

# Define the target directory relative to the project root
# We assume this script is run from the project root or we can adjust paths
TARGET_DIR = os.path.join("data", "raw", "ptbxl")

def download_ptbxl():
    print(f"Downloading PTB-XL dataset to {TARGET_DIR}...")
    
    # Create directory if it doesn't exist
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    try:
        # Download the database
        # This might take a while depending on internet connection
        wfdb.dl_database('ptb-xl', dl_dir=TARGET_DIR)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you have an internet connection and the 'wfdb' library installed.")

if __name__ == "__main__":
    download_ptbxl()
