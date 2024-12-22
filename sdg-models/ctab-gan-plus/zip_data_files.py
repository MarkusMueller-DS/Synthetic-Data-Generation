import os
import zipfile

output_zip_path = "data_files.zip"

file_paths = [
    "Real_Datasets/adult/train_src.csv",
    "Real_Datasets/adult/train.csv",
    "Real_Datasets/adult/test.csv",
    "Fake_Datasets/adult/Adult_fake_0.csv",
]

with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file_path in file_paths:
        # Verify file exists before adding
        if os.path.isfile(file_path):
            # Add file with its relative path for better organization
            arcname = os.path.basename(file_path)
            zipf.write(file_path, arcname=arcname)
            print(f"Added {file_path} to ZIP as {arcname}")
        else:
            print(f"File not found: {file_path}")

print(f"ZIP file created")
