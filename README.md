## Music Auto-tagging Using CNNs and Mel-spectrograms with Reduced Frequency and Time Resolution

The code in this repository was used to generate the results that are presented in the paper. Here we also include the complete **[tables](./results)** with the performance of all the combinations of settings that are not shown in the paper. Additionally, we show some **[examples](./demos)** of mel-spectrograms sonified to give an idea of the information that the network gets for each setting.

### Abstract

Automatic tagging of music is an important research topic in Music Information Retrieval and audio analysis algorithms proposed for this task have achieved improvements with advances in deep learning. In particular, many state-of-the-art systems use Convolutional Neural Networks and operate on mel-spectrogram representations of the audio. In this paper, we compare commonly used mel-spectrogram representations and evaluate model performances that can be achieved by reducing the input size in terms of both lesser amount of frequency bands and larger frame rates. We use the MagnaTagaTune dataset for comprehensive performance comparisons and then compare selected configurations on the larger Million Song Dataset. The results of this study can serve researchers and practitioners in their trade-off decision between accuracy of the models, data storage size and training and inference times.

### Contents of the repository

This repository contains the following folders and files:
 - **msd-tagging**: This folder contains the code to reproduce the experiments for the million song dataset
 - **mtat-tagging**: This folder contains the code to reproduce the experiments for the MagnaTagATune dataset
 - **[results](./results)**: This page contains the tables with the performance of the models for all the settings
 - **[Sonify.ipynb](./Sonify.ipynb)**: This python notebook has the code to sonify the mel-spectrograms
 - **[examples](./demos)**: This page contains some examples of sonified mel-spectrograms for different settings


### Acknowledgements

This work was partially supported by Kakao Corp.
