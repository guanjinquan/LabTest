import os
from torchvision.io import read_image
from skimage.transform import resize
import numpy as np
import torch


def another_site(file):
    if file.startswith("01"):
        return file.replace("01", "02")
    else:
        return file.replace("02", "01")


if __name__ == "__main__":
    # Don't change this line and the file position
    os.chdir(os.path.dirname(__file__) + "/../")  
    
    input_dir = "/home/Guanjq/HuangData/PartialPathologyImages"  # setting the path to the folder containing the pathology data
    output_dir = "./Data/NpyData"
    os.makedirs(output_dir, exist_ok=True)

    # resize the data into 512x512 and save them into npy files
    for dir_name in os.listdir(input_dir):
        pat_dir = os.path.join(input_dir, dir_name)
        
        try:
            pid = int(dir_name.split("_")[0])  # 只有数字pid是需要的
            print("pid: ", pid, flush=True)
        except:
            continue
        
        pid_str = dir_name.split("_")[0] + ".npy"  # pid.npy
        Data = None
        exist_image_set = set()
        missed_image_set = set()
        
        # some images are missing, we need to check if the corresponding image in another site exists
        for file in ["01_2X.jpg", "01_4X.jpg", "01_10X.jpg", "02_2X.jpg", "02_4X.jpg", "02_10X.jpg"]:
            img_path = os.path.join(pat_dir, file)
            if not os.path.exists(img_path):
                missed_image_set.add(file)
            else:
                exist_image_set.add(file) 
        
        if os.path.exists(os.path.join(output_dir, str(pid) + ".npy")):
            continue
        
        # if one site is missing, we use the image from another site
        for file in ["01_2X.jpg", "01_4X.jpg", "01_10X.jpg", "02_2X.jpg", "02_4X.jpg", "02_10X.jpg"]:
            if file in missed_image_set:
                if another_site(file) in exist_image_set:
                    file = another_site(file)
                else:
                    raise Exception("Both sites are missing")
            try:
                img = read_image(os.path.join(pat_dir, file))  # enusre the image is RGB and not Empty
            except Exception as e:
                img = torch.zeros((3, 512, 512))
                print("Error: ", e, "pid: ", pid, "file: ", file)
                
            img = resize(img.detach().numpy().astype(np.float32), (3, 512, 512))  # 如果不转成float，会导致输出结果在01之间

            # Data [6, 3, 512, 512]
            if Data is None:
                Data = np.expand_dims(img, axis=0)
            else:
                Data = np.concatenate((Data, np.expand_dims(img, axis=0)), axis=0)
        
        np.save(os.path.join(output_dir, pid_str), Data)  
    
