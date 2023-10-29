In this directory youll find the necessary files to train an ANIEFP style network with electrostatic potential as added input

the dataset is taken from the example directory in this repo

there are a handful of python files that I will explain

main_functions.py: a set of functions used in the training

make_data.py: a function that takes the csv and turns it into a bunch of torch tensors for training
    - note: this is a poor implementation of this, future iterations will use the pytorch dataset class, but that work hasnt been published yet
    - for now, this is just a proof of concept, so feel free to change things around and make them better, or just wait for me to release the better stuff 

template_ANIEFP_network.py: this is a template training file, most of the code comes from here: https://aiqm.github.io/torchani/examples/nnp_training.html
    which is the torchani tutorial on how to train your own neural net using torchani. 
    this template file has a few values omitted, mainly the places that say: #insert_wdecay1 #insert_wdecay2 #insert_lr_f #insert_name
    the reason for this is because these are usually a spread of values over multiple different networks, so you can change them to be whatever you want

template_ANIEFP_network_withvalues.py: this is the same as the previous template, but this time i've left in the actual values for one of the network in the weight decay and etc
    I will comment this file because this one can actually be run without errors
