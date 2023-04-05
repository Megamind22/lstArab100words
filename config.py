
args = dict()


#project structure
args["DEMO_DIRECTORY"] = "/content/project_test/demo"   #absolute path to the demo directory
args["TRAINED_MODEL_FILE"] = "/content/lstArab100words/Newfolder/pretrain_001w-step_0026-wer_0.185.pt" #relative path to the trained model file
args["TRAINED_FRONTEND_FILE"] = "/content/lstArab100words/Newfolder/visual_frontend.pt" #absolute path to the trained visual frontend file


#data
args["PRETRAIN_VAL_SPLIT"] = 0.01   #validation set size fraction during pretraining
args["NUM_WORKERS"] = 8 #dataloader num_workers argument
args["PRETRAIN_NUM_WORDS"] = 1  #number of words limit in current curriculum learning iteration
args["MAIN_REQ_INPUT_LENGTH"] = 145 #minimum input length while training
args["CHAR_TO_INDEX"] ={" ":1, "1":30, "0":29, 
        "ؤ":37,
         "إ":32, 
        "ء":34, 
        "ى":38,
        "ة":36,
        "ا":35 ,
        "ئ":22,
        "أ":5,
       "ب":20,
       "ت":17,
       "ث":12,
       "ج":2,
       "ح":19,
       "خ":16,
       "د":9,
       "ذ":6,
       "ر":25,
       "ز":24,
       "س":11,
       "ش":18,
       "ص":7,
       "ض":4,
       "ط":21,
       "ظ":27,
       "ع":10,
       "غ":8,
       "ف":3,
       "ق":13,
       "ك":23,
       "ل":15,
       "م":26,
       "ن":14,
       "ه":28,
       "و":8,
       "ي":9,
       "<EOS>":39,
            }      #ch
args["INDEX_TO_CHAR"] ={1:" ",30:"1", 29:"0", 
        37:"ؤ", 
        32:"إ", 
        34:"ء", 
        38:"ى", 
        36:"ة", 
        35:"ا",
        22:"ئ",
        5:"أ",
       20:"ب",
       17:"ت",
       12:"ث",
       2:"ج",
       19:"ح",
       16:"خ",
       9:"د",
       6:"ذ",
       25:"ر",
       24:"ز",
       11:"س",
       18:"ش",
       7:"ص",
       4:"ض",
       21:"ط",
       27:"ظ",
       10:"ع",
       8:"غ",
       3:"ف",
       13:"ق",
       23:"ك",
       15:"ل",
       26:"م",
       14:"ن",
       28:"ه",
       8:"و",
       9:"ي",
       39:"<EOS>",
            } 
#preprocessing
args["VIDEO_FPS"] = 30  #frame rate of the video clips
args["ROI_SIZE"] = 112  #height and width of input greyscale lip region patch
args["NORMALIZATION_MEAN"] = 0.4161 #mean value for normalization of greyscale lip region patch
args["NORMALIZATION_STD"] = 0.1688  #standard deviation value for normalization of greyscale lip region patch


#training
args["SEED"] = 19220297 #seed for random number generators
args["BATCH_SIZE"] = 32 #minibatch size
args["STEP_SIZE"] = 69   #number of samples in one step (virtual epoch)
args["NUM_STEPS"] = 34 #maximum number of steps to train for (early stopping is used)
args["SAVE_FREQUENCY"] = 17 #saving the model weights and loss/metric plots after every these many steps


#optimizer and scheduler
args["INIT_LR"] = 1e-4  #initial learning rate for scheduler
args["FINAL_LR"] = 1e-6 #final learning rate for scheduler
args["LR_SCHEDULER_FACTOR"] = 0.5   #learning rate decrease factor for scheduler
args["LR_SCHEDULER_WAIT"] = 25  #number of steps to wait to lower learning rate
args["LR_SCHEDULER_THRESH"] = 0.001 #threshold to check plateau-ing of wer
args["MOMENTUM1"] = 0.9 #optimizer momentum 1 value
args["MOMENTUM2"] = 0.999   #optimizer momentum 2 value


#model
args["NUM_CLASSES"] = 40    #number of output characters


#transformer architecture
args["PE_MAX_LENGTH"] = 2500    #length up to which we calculate positional encodings
args["TX_NUM_FEATURES"] = 512   #transformer input feature size
args["TX_ATTENTION_HEADS"] = 8  #number of attention heads in multihead attention layer
args["TX_NUM_LAYERS"] = 6   #number of Transformer Encoder blocks in the stack
args["TX_FEEDFORWARD_DIM"] = 2048   #hidden layer size in feedforward network of transformer
args["TX_DROPOUT"] = 0.1    #dropout probability in the transformer


#beam search
args["BEAM_WIDTH"] = 100    #beam width
args["LM_WEIGHT_ALPHA"] = 0.5   #weight of language model probability in shallow fusion beam scoring
args["LENGTH_PENALTY_BETA"] = 0.1   #length penalty exponent hyperparameter
args["THRESH_PROBABILITY"] = 0.0001 #threshold probability in beam search algorithm
args["USE_LM"] = False  #whether to use language model for decoding


#testing
args["TEST_DEMO_DECODING"] = "greedy"   #test/demo decoding type - "greedy" or "search"


if __name__ == "__main__":

    for key,value in args.items():
        print(str(key) + " : " + str(value))
