#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""


# In[5]:


import torch
import os
from tqdm import tqdm
import numpy as np

from config import args
from visual_frontend import VisualFrontend
from preprocessing import preprocess_sample


# In[6]:


def main():

    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")



    #declaring the visual frontend module
    vf = VisualFrontend()
    vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"], map_location=device))
    vf.to(device)


    #walking through the data directory and obtaining a list of all files in the dataset
    filesList = list()
    for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
        for file in files:
            if file.endswith(".mp4"):
                filesList.append(os.path.join(root, file[:-4]))


    #Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" %(len(filesList)))
    print("\n\nStarting preprocessing ....\n")

    params = {"roiSize":args["ROI_SIZE"], "normMean":args["NORMALIZATION_MEAN"], "normStd":args["NORMALIZATION_STD"], "vf":vf}
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file,vf)

    print("\nPreprocessing Done.")

    
    #Generating preval.txt for splitting the pretrain set into train and validation sets
    print("\n\nGenerating the preval.txt file ....")

    with open(args["DATA_DIRECTORY"] + "\\train.txt", "r") as f:
        lines = f.readlines()

    if os.path.exists(args["DATA_DIRECTORY"] + "\\preval.txt"):
        with open(args["DATA_DIRECTORY"] + "\\val.txt", "r") as f:
            lines.extend(f.readlines())

    indices = np.arange(len(lines))
    np.random.shuffle(indices)
    valIxs = np.sort(indices[:int(np.ceil(args["PRETRAIN_VAL_SPLIT"]*len(indices)))])
    trainIxs = np.sort(indices[int(np.ceil(args["PRETRAIN_VAL_SPLIT"]*len(indices))):])

    lines = np.sort(np.array(lines))
    with open(args["DATA_DIRECTORY"] + "\\train.txt", "w") as f:
        f.writelines(list(lines[trainIxs]))
    with open(args["DATA_DIRECTORY"] + "\\val.txt", "w") as f:
        f.writelines(list(lines[valIxs]))

    print("\npreval.txt file generated.\n")

    return


# In[7]:


main()


# In[ ]:





# In[ ]:




