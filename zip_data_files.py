import os
import shutil


folder_to_zip = "data/synthetic"
zip_folder_name = "syn_data"

shutil.make_archive(zip_folder_name, "zip", folder_to_zip)

print(f"Created zip folder: {zip_folder_name}")
