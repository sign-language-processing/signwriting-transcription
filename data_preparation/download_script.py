# save the following code in a file, e.g., /content/downloadscript.py

import requests
import sys

def download_file(url, destination_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
        print(f"Download successful. File saved as {destination_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python /content/downloadscript.py <URL> <destination_path>")
        sys.exit(1)

    url = sys.argv[1]
    destination_path = sys.argv[2]

    download_file(url, destination_path)
