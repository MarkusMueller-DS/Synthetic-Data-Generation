import shutil

# Path to the folder to be zipped
folder_to_zip = "data/synthetic"

# Output path for the zipped file (without extension)
output_zip = "syn_data"

# Create the zip archive
shutil.make_archive(output_zip, "zip", folder_to_zip)
