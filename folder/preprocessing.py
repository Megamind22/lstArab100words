import numpy as np
import torch
import time
import cv2 as cv
from dataloader.dataloader import AVSRDataLoader
from tracker.face_tracker import FaceTracker
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")

def preprocess_sample(data_filename , params):

    videoFile=data_filename+".mp4"
    roiFile=data_filename+".png"
    file_save  = data_filename+".npy"

    normMean = params["normMean"]
    normStd = params["normStd"]
    vf = params["vf"]
  
    print(data_filename)
    dl=AVSRDataLoader(
                modality="video",
                speed_rate=30/25,#frameRate
                disable_transform=True
            )
    face_tracker=FaceTracker(device="cuda")
    end = time.time()
    landmarks = face_tracker.tracker(videoFile)
    print(f"face tracking speed: {len(landmarks) / (time.time() - end):.2f} fps.")
    sequence =dl.load_data(
        videoFile,
        landmarks,
    )
    sequence=sequence/255
    cv.imwrite(roiFile, np.floor(255 * np.concatenate(sequence, axis=1)).astype(np.int))
    inp=np.stack(sequence,axis=0)
    #print(inp)
    inp = np.expand_dims(inp, axis=[1])
    inp = (inp - normMean) / normStd
    # conver the array to tenser

    np.save(file_save, inp)


