import numpy as np
import pandas as pd 
import cv2 
import glob

def load_data(Infected_path, Uninfected_path):
    paths = [Infected_path, Uninfected_path]
    images = []
    labels = []
    for path in paths:
        path2 = path + "/*.png"
        for file in glob.glob(path2):
            # print(file)
            img = cv2.imread(file)
            if img is not None:
                resized_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)  # Resize the image
                images.append(resized_img)
                if path == Infected_path:
                    labels.append(0)  # Label for Infected
                else:
                    labels.append(1)  # Label for Uninfected
    return np.array(images), np.array(labels) 

if __name__ == "__main__":
    Infected_path = "cell_images/cell_images/Parasitized"
    Uninfected_path = "cell_images/cell_images/Uninfected"
    images, labels = load_data(Infected_path, Uninfected_path)
    print(f"Loaded {len(images)} images with labels.") 
    print(f"Image shape: {images[0].shape}")
    


