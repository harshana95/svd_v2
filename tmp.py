import os
import glob
import shutil 

path = "/depot/chan129/users/harshana/Datasets/long exposure time dataset"
dest = "/depot/chan129/users/harshana/Datasets/long_exposure_dataset"

folders = glob.glob(os.path.join(path, "*"))
for folder in folders:
    src = os.path.join(folder, "Camera_0.png")
    dst = os.path.join(dest, "gt", os.path.basename(folder)+".png")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)

    src = os.path.join(folder, "Camera_1.png")
    dst = os.path.join(dest, "5x-color", os.path.basename(folder)+".png")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)

    src = os.path.join(folder, "Camera_2.png")
    dst = os.path.join(dest, "5x-mono", os.path.basename(folder)+".png")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
