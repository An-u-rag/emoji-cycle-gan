# emoji-cycle-gan

A simplified implementation of Vanilla GAN (LSGAN) and Cycle GAN for image translation between windows and apple emojis. 


# Options.py
This file has all the relevant options to control the training and testing of the cycle and vanilla GANS with some base options. 
Specifically 2 options which are "format" and "d_sigmoid". 
"format" options is used for selecting the type of input we want to handle that is "RBG" or "RGBA", the rest of the code automatically accounts for the changes in the model. 
The "d_sigmoid" is a boolean option which is used to apply a sigmoid activation function at the end of the discriminator. However according to tests, better results visually are obtained using sigmoid. 

Note: The default Discrimiantor model uses LeakyReLU() after each conv_layer since it gave better results. 
However there is an alternate discriminator model called "NoLeakDiscriminator" in models.py which uses just ReLU() activation function in discriminator instead of LeakyReLU().
