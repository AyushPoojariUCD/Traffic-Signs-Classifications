import os
import zipfile
from tqdm import tqdm

def extract_zip_file(zip_path, extract_to=None):

    if extract_to is None:
        base_name = os.path.basename(zip_path)
        folder_name = os.path.splitext(base_name)[0]
        extract_to = os.path.join(os.path.dirname(zip_path), folder_name)

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()

        for file in tqdm(file_list, desc=f"Extracting to '{extract_to}'"):
            zip_ref.extract(member=file, path=extract_to)

    print(f"\n Extracted to '{extract_to}' folder")

    print("\n Folder contents:")
    for item in os.listdir(extract_to):
        print(f"- {item}")
