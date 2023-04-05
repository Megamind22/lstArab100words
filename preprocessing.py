import numpy as np
import torch
import time
import cv2 as cv
from dataloader.dataloader import AVSRDataLoader
from tracker.face_tracker import FaceTracker

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
    inp = np.expand_dims(inp, axis=[1, 2])
    inp = (inp - normMean) / normStd
    # conver the array to tenser
    inputBatch = torch.from_numpy(inp)
    inputBatch = (inputBatch.float()).to("cuda")
    #print(inputBatch.shape)
    vf.eval()
    with torch.no_grad():
        outputBatch = vf(inputBatch)
    out = torch.squeeze(outputBatch, dim=1)
    out=out.cpu().numpy()
    print(out.shape)
    np.save(file_save, out)


