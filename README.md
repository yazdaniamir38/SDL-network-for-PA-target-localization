# SDL-network-for-PA-target-localization
Following packages and libraries are needed for running the codes:

Python 3.6

pytorch 1.1.0

dsntnn 0.5.3

scipy  2.9.0

h5py

'Train.py' is used for taining different architecture and losses by using the 'Mode' variable:

'SDL':SDL architecture 

'WN-less' without wavefront/noise filters (but with the denoising decoder'

'single' single autoencoder

##The netowrk can predict up to 4 targets in each sample and it outputs (0,0) for no detection.

In order to test the network use 'test.py' providing the following parameters:

'checkpoint':checkpoint directory

'test_dir':test set directory

'train'directory' includes h5 files for the training set.

make sure the test set follows 'NCWH' for the dimensions.

The dataset and pretrained models can be found here:

https://drive.google.com/drive/folders/1tPXmnqvpFsgJvs68_2o3fIAJY6YBhTGA?usp=sharing

