import os
import shutil
from PIL import Image
import numpy as np

def images_equal(img1_path, img2_path, tolerance=1) -> bool:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        if img1.size != img2.size or img1.mode != img2.mode:
            return False
        
        img1_array = np.array(img1).astype(np.int16)
        img2_array = np.array(img2).astype(np.int16)
        
        diff = np.mean(img1_array - img2_array)
        return (diff <= tolerance)

def image_directories_equal(dir1, dir2) -> bool:
    dir1_files = sorted(os.listdir(dir1))
    dir2_files = sorted(os.listdir(dir2))

    if dir1_files != dir2_files:
        return False

    for file_name in dir1_files:
        file1_path = os.path.join(dir1, file_name)
        file2_path = os.path.join(dir2, file_name)

        if os.path.isdir(file1_path) and os.path.isdir(file2_path):
            if not image_directories_equal(file1_path, file2_path):
                return False
        elif os.path.isfile(file1_path) and os.path.isfile(file2_path):
            if not images_equal(file1_path, file2_path):
                return False
        else:
            return False

    return True

class WorkspaceDirectory:
    def __init__(self, workspace_dir, subdirs:list=[]):
        self.workspace_dir = workspace_dir
        try:
            shutil.rmtree(self.workspace_dir)
        except FileNotFoundError:
            pass

        for subdir in subdirs:
            os.makedirs(os.path.join(workspace_dir, subdir))

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            shutil.rmtree(self.workspace_dir)
        except FileNotFoundError:
            print(f"Directory {self.workspace_dir} not found. Although it should have been created.")