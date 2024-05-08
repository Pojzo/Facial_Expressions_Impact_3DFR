import os
import shutil
import glob
from PIL import Image

if os.path.exists('neutral_images'):
    try: 
        os.rmdir('neutral_images')
        print("Directory removed")
    except: 
        print("Could not remove directory")
        exit()

os.mkdir('neutral_images')

input_path = './'

# search recursively in cfd_image for .jpg files that end with -N

files = []

ratio = 2444 / 1718

for file in glob.glob(input_path + '/**/*-N.jpg', recursive=True):
    # resize image to 450x450
    with Image.open(file) as img:
        img = img.resize((450,  int(450 / ratio)))
        img.save('neutral_images/' + os.path.basename(file))
    files.append('neutral_images/' + os.path.basename(file))

print(files)