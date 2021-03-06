# sc2GAN-FE
Hello everyone,
This is the related code for our work 'sc2GAN-FE'.
Let me teach you how to use it!

Firstly, you need to download datasets, their names are '92av3c_145x145_200.mat', 'paviaU_300x145_103.mat', 'salinas_512x217_224.mat', and 'DFC_900x600_50.mat'.
The datasets have been updated to https://doi.org/10.5281/zenodo.4587297

Then, run codes in the 'data prepossessing' file.
1. step1.py
2. step2.m

Third, run 'main.py' to train the network!

After that, you will obtain a latent embedding, you can use your SVM codes or our codes in the 'SVM classification' file.
If the latter, run 'SVMclassification.m'!

Codes for all datasets are provided. If you want to use other datasets, you need to change some related parameters, like the shape of data, the number of labels, etc. These parameters are quite clear for you to find, so it is a easy work to change codes. 

Moreover, comments and explanations are added in the codes for the 92av3c datasets. Good luck!!!!!!!

