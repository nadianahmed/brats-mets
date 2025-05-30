import os
import gzip
import shutil

base_dir = '/project/def-sreeram/hsheikh1/brats-mets/Datasets/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData'

for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.gz'):
            gz_path = os.path.join(root, file)
            nii_path = os.path.splitext(gz_path)[0]  # remove .gz
            if not os.path.exists(nii_path):
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(nii_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print("Unzipped:", gz_path)
