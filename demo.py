
import torch
import numpy as np
import cv2 as cv
import os

from config import args
from models.video_net import VideoNet
from models.visual_frontend import VisualFrontend
from models.lrs2_char_lm import LRS2CharLM
from utils import prepare_main_input, collate_fn
from preprocessing import preprocess_sample
from decoders import ctc_greedy_decode, ctc_search_decode



def main():

    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")


    if args["TRAINED_MODEL_FILE"] is not None:

        print("\nTrained Model File: %s" %(args["TRAINED_MODEL_FILE"]))
        print("\nDemo Directory: %s" %(args["DEMO_DIRECTORY"]))


        #declaring the model and loading the trained weights
        model = VideoNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
                         args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
        model.load_state_dict(torch.load(args["TRAINED_MODEL_FILE"], map_location=device))
        model.to(device)


        #declaring the visual frontend module
        vf = VisualFrontend()
        vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"], map_location=device))
        vf.to(device)


        #declaring the language model
        if not args["USE_LM"]:
            lm = None
        else:
          lm = LRS2CharLM()
          lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"], map_location=device))
          lm.to(device)
        


        print("\n\nRunning Demo .... \n")

        #walking through the demo directory and running the model on all video files in it
        for root, dirs, files in os.walk(args["DEMO_DIRECTORY"]):
            for file in files:
                if file.endswith(".mp4"):
                    sampleFile = os.path.join(root, file[:-4])

                    #preprocessing the sample
                    params = {"roiSize":args["ROI_SIZE"], "normMean":args["NORMALIZATION_MEAN"], "normStd":args["NORMALIZATION_STD"], "vf":vf}
                    preprocess_sample(sampleFile, params)

                    #converting the data sample into appropriate tensors for input to the model
                    visualFeaturesFile = os.path.join(root, file[:-4]) + ".npy"
                    videoParams = {"videoFPS":args["VIDEO_FPS"]}
                    inp, _, inpLen, _ = prepare_main_input(visualFeaturesFile, None, args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"],
                                                           videoParams)
                    inputBatch, _, inputLenBatch, _ = collate_fn([(inp, None, inpLen, None)])

                    #running the model
                    inputBatch = (inputBatch.float()).to(device)
                    inputLenBatch = (inputLenBatch.int()).to(device)
                    model.eval()
                    with torch.no_grad():
                        outputBatch = model(inputBatch)

                    #obtaining the prediction using CTC deocder
                    if args["TEST_DEMO_DECODING"] == "greedy":
                        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch, args["CHAR_TO_INDEX"]["<EOS>"])

                    elif args["TEST_DEMO_DECODING"] == "search":
                        beamSearchParams = {"beamWidth":args["BEAM_WIDTH"], "alpha":args["LM_WEIGHT_ALPHA"], "beta":args["LENGTH_PENALTY_BETA"],
                                            "threshProb":args["THRESH_PROBABILITY"]}
                        predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch, beamSearchParams,
                                                                                args["CHAR_TO_INDEX"][" "], args["CHAR_TO_INDEX"]["<EOS>"], lm)

                    else:
                        print("Invalid Decode Scheme")
                        exit()

                    #converting character indices back to characters
                    pred = predictionBatch[:][:-1]
                    pred = "".join([args["INDEX_TO_CHAR"][ix] for ix in pred.tolist()])

                    #printing the predictions
                    print("File: %s" %(file))
                    print("Prediction: %s" %(pred))
                    print("\n")


        print("Demo Completed.\n")


    else:
        print("\nPath to trained model file not specified.\n")

    return



if __name__ == "__main__":
    main()
