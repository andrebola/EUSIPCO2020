# Million Song Dataset experiments

In this folder is located the code to train and evaluate the tagging on MSD dataset. These are the files located in this foder:

 - audio_extract.py: This file contains the code to extract the melspectograms from the audio according to the given parameters
 - generate_mels: This script calls 'audio_extract.py' and generates all the melspectograms given a folder path
 - autotagging_vgg.py trains and evalutates the VGG model on when reducing frquency resolution for 12000 SR
 - autotagging_vgg_16.py trains and evalutates the VGG model on when reducing frquency resolution for 16000 SR
 - autotagging_vgg_time.py trains and evalutates the VGG model on when reducing time resolution (only x2) for 12000 SR
 - autotagging_vgg_time_16.py trains and evalutates the VGG model on when reducing time resolution (only x2) for 16000 SR
 - autotagging.py trains and evalutates the MUSICNN model

