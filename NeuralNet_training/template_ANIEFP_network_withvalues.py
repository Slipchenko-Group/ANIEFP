from make_data import  make_data
from main_functions import validate
import torch
import torch.nn.functional as F
import torchani
import torch.utils.tensorboard
import numpy as np
import os

# Most of this code comes from the torchani tutorial that I linked in the readme
# I will only explain the extra stuff that is added, for everything else please refer to that tutorial

# here we call the make_data function in order to put our dataset csv into torch tensors to use for training
# this is a very workaround way of doing so and was done before I learned about pytorch dataset classes
# this method works though and is fine, but will not work well with adding forces to training
# in the future, I use dataset classes and that will remove almost all of this extra code
# but for now this works
# the label_col variable is the value we are going to train to, you can change that to be anything
species, cspecies, aevs_elecpots, energies, mol_names = make_data(label_col='corr_energy', subtractE = 0, csv_file='dataset')


# device is a standard pytorch thing
# just choosing between running on cpu or gpu
device = energies.device

# here we take our species and aevs and energies etc etc and do a 80:20 random split for testing and validation
# in this case there is no 3 way split for train/test/val
# in the future I implement this, but not here
# you are free to split the data in any way you wish
# the test set is really the validation set, it is used to evaluate the network on untrained data and save a "best" set of parameters
# will be explained later
np.random.seed(0)

p80 = int(energies.shape[0]*0.8)
total_pts = np.arange(0, int(energies.shape[0]), dtype='int32')
np.random.shuffle(total_pts)

np.random.seed(0)
p80 = int(energies.shape[0]*0.8)
total_pts = np.arange(0, int(energies.shape[0]), dtype='int32')
np.random.shuffle(total_pts)

train_cspecies = cspecies[total_pts[0:p80]]
train_species = species[total_pts[0:p80]]
train_aep = aevs_elecpots[total_pts[0:p80]]
train_energies = energies[total_pts[0:p80]].float()
train_names = [mol_names[i] for i in total_pts[0:p80]]

test_cspecies = cspecies[total_pts[p80:]]
test_species = species[total_pts[p80:]]
test_aep = aevs_elecpots[total_pts[p80:]]
test_energies = energies[total_pts[p80:]].float()
test_names = [mol_names[i] for i in total_pts[p80:]]


# so the ANIEFP network is just the ANI network with 1 extra node on each layer
# first we need to get the original ANI1 network
# so we first load the model and send it to the device/gpu
model_ANI1x = torchani.models.ANI1x(periodic_table_index=True).to(device)
# then we only grab the first one, ANI1x is actually an ensemble of networks that work together, we just want the first one
model_ANI1x_0 = model_ANI1x[0]

# these are values to set for how our training will go
max_epochs = 5000
early_stopping_learning_rate = 1.0E-5
wdecay1 = 0.0001
wdecay2 = 0.0001
lr_f = 0.5
name = "template_ANIEFP_network_withvalues"
# the size of our input vector
# so ANI1 is originally 384, and we add another node so its 385
aev_dim = 385
set_0bias = 1
add_0w = 0

# this is implementing the individual atom networks just like ANI1, but we add one to each layer
H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 161),
    torch.nn.CELU(0.1),
    torch.nn.Linear(161, 129),
    torch.nn.CELU(0.1),
    torch.nn.Linear(129, 97),
    torch.nn.CELU(0.1),
    torch.nn.Linear(97, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 145),
    torch.nn.CELU(0.1),
    torch.nn.Linear(145, 113),
    torch.nn.CELU(0.1),
    torch.nn.Linear(113, 97),
    torch.nn.CELU(0.1),
    torch.nn.Linear(97, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 129),
    torch.nn.CELU(0.1),
    torch.nn.Linear(129, 113),
    torch.nn.CELU(0.1),
    torch.nn.Linear(113, 97),
    torch.nn.CELU(0.1),
    torch.nn.Linear(97, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 129),
    torch.nn.CELU(0.1),
    torch.nn.Linear(129, 113),
    torch.nn.CELU(0.1),
    torch.nn.Linear(113, 97),
    torch.nn.CELU(0.1),
    torch.nn.Linear(97, 1)
)


# the following functions are all used for setting up the new ANIEFP network and making sure the weights and biases are correct for each layer

def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)
# here we make our network called net, with all the sub networks in it
net = torchani.ANIModel([H_network, C_network, N_network, O_network])
net.apply(init_params)

def add_zeros(model, ref_model, atom, param, device=device):
    params = []
    atom_dict = {'H' : 0, 'C' : 1, 'N' : 2, 'O' : 3}
    for i in [0, 2, 4, 6]:
        if param == 'weight':
            diff_units = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'].shape[1]-ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}'].shape[1]
        else:
            diff_units = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'].shape[0]-ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}'].shape[0]
        ptensor = ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}']
        if param == 'weight' and i < 6 :
            ptensor = torch.cat((ptensor, torch.zeros(ptensor.shape[0], diff_units).to(device)), -1).to(device)
        params.append(ptensor)
    return(params)

def add_vals(model, ref_model, atom, param, device=device):
    params = []
    atom_dict = {'H' : 0, 'C' : 1, 'N' : 2, 'O' : 3}
    for i in [0, 2, 4, 6]:
        if param == 'weight':
            diff_units = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'].shape[1]-ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}'].shape[1]
        else:
            diff_units = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'].shape[0]-ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}'].shape[0]
        ptensor = ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}']
        if param == 'weight' and i < 6 :
            new_vals = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'][0:ptensor.shape[0],ptensor.shape[1]:]
            ptensor = torch.cat((ptensor.to(device), new_vals.to(device)), -1).to(device)
        params.append(ptensor)
    return(params)

# this function will be used later to reset the ani portion of the network
def set_params(model, ref_model, atom, param, setB0, add0):
    atom_dict = {'H' : 0, 'C' : 1, 'N' : 2, 'O' : 3}
    if add0 == 1:
        l0, l2, l4, l6 = add_zeros(model, ref_model, atom, param)
    else:
        l0, l2, l4, l6 = add_vals(model, ref_model, atom, param)
    layer_params = [l0, l2, l4, l6]
    layer_idx = [0, 2, 4]
    for ii in range(len(layer_idx)):
        i = layer_idx[ii]
        units_diff = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'].shape[0]-ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}'].shape[0]
        if setB0 == 1 and param=='bias':
            model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'][...] = 0
        model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'][0:-units_diff,...] = layer_params[ii]
    if param == 'weight':
        units_diff = model.state_dict()[f'{atom_dict[atom]}.6.{param}'].shape[1]-ref_model.state_dict()[f'neural_networks.{atom}.6.{param}'].shape[1]
        model.state_dict()[f'{atom_dict[atom]}.6.{param}'][...,0:-units_diff] = l6
    else:
        model.state_dict()[f'{atom_dict[atom]}.6.{param}'][:] = l6
        

# so we take our 'net' and then take all the weights and biases from ANI1 and set them to the associated nodes for our net
# because we randomly initialized them first, and now we need to make sure that our network looks like ani1 + new node
set_params(net, model_ANI1x_0, 'H', 'bias', set_0bias, add_0w)
set_params(net, model_ANI1x_0, 'H', 'weight', 0, add_0w)
set_params(net, model_ANI1x_0, 'C', 'bias', set_0bias, add_0w)
set_params(net, model_ANI1x_0, 'C', 'weight', 0, add_0w)
set_params(net, model_ANI1x_0, 'N', 'bias', set_0bias, add_0w)
set_params(net, model_ANI1x_0, 'N', 'weight', 0, add_0w)
set_params(net, model_ANI1x_0, 'O', 'bias', set_0bias, add_0w)
set_params(net, model_ANI1x_0, 'O', 'weight', 0, add_0w)

# setting the networks parameters and weight decays 
# all from torchani tutorial
# using AdamW for weights, and SGD for biases
model_net = torchani.nn.Sequential(net).to(device)
AdamW = torch.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[2].weight], 'weight_decay': wdecay1},
    {'params': [H_network[4].weight], 'weight_decay': wdecay2},
    {'params': [H_network[6].weight]},
    # C networks
    {'params': [C_network[0].weight]},
    {'params': [C_network[2].weight], 'weight_decay': wdecay1},
    {'params': [C_network[4].weight], 'weight_decay': wdecay2},
    {'params': [C_network[6].weight]},
    # N networks
    {'params': [N_network[0].weight]},
    {'params': [N_network[2].weight], 'weight_decay': wdecay1},
    {'params': [N_network[4].weight], 'weight_decay': wdecay2},
    {'params': [N_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[2].weight], 'weight_decay': wdecay1},
    {'params': [O_network[4].weight], 'weight_decay': wdecay2},
    {'params': [O_network[6].weight]},
])
SGD = torch.optim.SGD([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].bias]},
    # C networks
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].bias]},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].bias]},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].bias]},
], lr=1e-3)

# checkpoint stuff, if there is a checkpoint file we can/will load it and start from there
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=lr_f, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=lr_f, patience=100, threshold=0)

latest_checkpoint = f'{name}_latest.pt'
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    net.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])

tensorboard = torch.utils.tensorboard.SummaryWriter()

mse = torch.nn.MSELoss(reduction='none')
best_model_checkpoint = f'{name}_best.pt'
best_rmse = 10000000000
updated_lr = 0


# this is the training loop
for _ in range(max_epochs):
    # first we evaluate the model on the training and "test" set
    # its a validation set but ill just refer to it as the test set because it was named that way
    rmse = validate(model_net, train_cspecies, train_aep, train_energies)
    rmse2 = validate(model_net, test_cspecies, test_aep, test_energies)
    # we then print it
    # the way i currently run these networks is i have a bash script that pipes the output to a text file
    # something like network_script.py > network_script.txt
    # just so i can look at the rmse's afterwards and evaluate it/plot it
    print(f'{_},{rmse},{rmse2}')
    learning_rate = AdamW.param_groups[0]['lr']
    if learning_rate < early_stopping_learning_rate:
        break
    if rmse < 5 and updated_lr == 0:
        AdamW.param_groups[0]['lr'] = learning_rate/10
        updated_lr = 1
        learning_rate = AdamW.param_groups[0]['lr']

    # checkpoint
    # here we do a validation step
    # we check to see if the test rmse is better than our current best, and also if the difference between our test and train is lest than 0.5
    # this is to prevent the best network being an overfitted network
    if rmse2 < best_rmse and abs(rmse-rmse2)<0.5:
    #if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(net.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)
    SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    # here we evaluate the training data
    # predict energies, calculate loss, zero the gradients, the do a backprop
    num_atoms = (train_cspecies >= 0).sum(dim=1, dtype=train_energies.dtype)
    _, predicted_energies = model_net((train_cspecies, train_aep))
    loss = (mse(predicted_energies, train_energies) / num_atoms.sqrt()).mean()
    AdamW.zero_grad()
    SGD.zero_grad()
    loss.backward()
    for child in model_net.children():
        for pname, param in child.named_parameters():
            if 'weight' in pname and '6.weight' not in pname:
                param.grad[0:-1,:] = 0
            elif 'bias' in pname and '6.bias' not in pname:
                param.grad[0:-1] = 0
            elif '6.weight' in pname:
                param.grad[...,0:-1] = 0
            elif '6.bias' in pname:
                param.grad[:] = 0

    # we have to reset the ANI part of the network to be the original ani values then step
    set_params(net, model_ANI1x_0, 'H', 'bias', set_0bias, add_0w)
    set_params(net, model_ANI1x_0, 'H', 'weight', 0, add_0w)
    set_params(net, model_ANI1x_0, 'C', 'bias', set_0bias, add_0w)
    set_params(net, model_ANI1x_0, 'C', 'weight', 0, add_0w)
    set_params(net, model_ANI1x_0, 'N', 'bias', set_0bias, add_0w)
    set_params(net, model_ANI1x_0, 'N', 'weight', 0, add_0w)
    set_params(net, model_ANI1x_0, 'O', 'bias', set_0bias, add_0w)
    set_params(net, model_ANI1x_0, 'O', 'weight', 0, add_0w)
    AdamW.step()
    SGD.step()

    # write current batch loss to TensorBoard
    tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(train_energies))

    torch.save({
        'nn': net.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)


