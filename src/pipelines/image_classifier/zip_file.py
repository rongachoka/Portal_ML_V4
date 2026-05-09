"""
zip_file.py
===========
Google Colab utility: zips the Brand_Images_TopDown training dataset from Drive.

Mounts Google Drive in a Colab runtime, zips the image training folder, and
saves the archive to Drive for transfer or backup.

Input:  Google Drive: /Portal_ML/.../Brand_Images_TopDown/
Output: Google Drive: Brand_Images_TopDown.zip

Run inside Google Colab before training on a new environment.
"""

import os
import shutil
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# The folder you want to zip
SOURCE_DIR = (
    '/content/drive/MyDrive/Portal_ML/Portal_ML_V4/data (1)/01_raw (1)/'
    'Brand_Images_TopDown'
)

# The name of the zip file we are creating (Do NOT add .zip to the end here)
TARGET_ZIP = (
    '/content/drive/MyDrive/Portal_ML/Portal_ML_V4/data (1)/01_raw (1)/'
    'Brand_Images_TopDown_Zipped'
)

print(f"📦 Zipping {SOURCE_DIR}...")
print("⏳ This might take a few minutes since Drive is doing the heavy lifting...")

# shutil.make_archive handles the zipping process
shutil.make_archive(TARGET_ZIP, 'zip', SOURCE_DIR)

print("✅ Zipping complete! You now have a permanent .zip file in your Drive.")