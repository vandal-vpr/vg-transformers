import tarfile
import glob
from datetime import datetime
import os
import shutil
import zipfile

DOWNLOAD_DIR = 'downloaded'
EXTRACT_TO = '/home/gabrielet/datasets/robotcar_val'

files = sorted(glob.glob('downloaded/*.tar'))
for file_index, file in enumerate(files):
    print(f"{str(datetime.now())[11:19]}  {file_index:02d} / {len(files)}  {file}")
    tar = tarfile.open(file)
    tar.extractall('temp/')
    tar.close()

os.makedirs(EXTRACT_TO, exist_ok=True)
os.makedirs(os.path.join(EXTRACT_TO, 'meta'))
with zipfile.ZipFile(os.path.join('downloaded', 'meta.zip'), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(EXTRACT_TO, 'meta'))

for untarred in os.listdir('temp'):
    shutil.move(os.path.join('temp', untarred), EXTRACT_TO)
shutil.rmtree('temp')
