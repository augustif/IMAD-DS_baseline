import os
import requests
from tqdm import tqdm
import py7zr
import numpy as np

def download_file(url, local_filename):
    # Stream the download to avoid loading the entire file into memory
    with requests.get(url, stream=True, verify=False) as r:
        r.raise_for_status()
        # Get the total file size from the headers
        total_size = int(r.headers.get('content-length', 0))
        # Open the local file for writing in binary mode
        with open(local_filename, 'wb') as f:
            # Use tqdm to display a progress bar
            for chunk in tqdm(
                    r.iter_content(
                        chunk_size=8192),
                    total=total_size // 8192,
                    unit='KB',
                    desc='Downloading file {url}'):
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
    print(f"Download completed: {local_filename}")


def unzip_7z_file(file_path, extract_to):
    # Ensure the destination directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Open the .7z file and extract its contents
    with py7zr.SevenZipFile(file_path, mode='r') as archive:
        archive.extractall(path=extract_to)

    print(f"Extraction completed: {file_path} to {extract_to}")
    # Remove the .7z file after extraction
    os.remove(file_path)
    print(f"Removed the .7z file: {file_path}")