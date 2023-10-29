In this directory youll find the necessary files to train an ANIEFP style network with electrostatic potential as added input \n
the dataset is taken from the example directory in this repo\n
there are a handful of python files that I will explain\n
main_functions.py: a set of functions used in the training\n
make_data.py: a function that takes the csv and turns it into a bunch of torch tensors for training\n
    - note: this is a poor implementation of this, future iterations will use the pytorch dataset class, but that work hasnt been published yet\n
    - for now, this is just a proof of concept, so feel free to change things around and make them better, or just wait for me to release the better stuff \n
template_ANIEFP_network.py: this is a template training file, most of the code comes from here: https://aiqm.github.io/torchani/examples/nnp_training.html\n
    which is the torchani tutorial on how to train your own neural net using torchani. \n
    this template file has a few values omitted, mainly the places that say: #insert_wdecay1 #insert_wdecay2 #insert_lr_f #insert_name\n
    the reason for this is because these are usually a spread of values over multiple different networks, so you can change them to be whatever you want\n
template_ANIEFP_network_withvalues.py: this is the same as the previous template, but this time i've left in the actual values for one of the network in the weight decay and etc\n
    I will comment this file because this one can actually be run without errors\n
