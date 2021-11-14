import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import time
import sys
sys.path.insert(1,'../src')
from models import ConvLSTM,PhyCell, EncoderRNN, EncoderDecoderRNN
from moving_mnist import MovingMNIST
from constrain_moments import K2M
# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
import argparse
from visualization import plot_spatio_temporal_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--nepochs', type=int, default=20, help='nb of epochs')
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=10, help='')
parser.add_argument('--save_name', type=str, default='phydnet', help='')
args = parser.parse_args()




root = './'
n_frames = 20
num_digits = 2
image_size = 64
digit_size = 28
N = 4 # total number of samples including training and validation data
mask = np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,0])
#mask = np.array([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0])
mm = MovingMNIST(root,
                 n_frames,
                 mask,
                 num_digits,
                 image_size,
                 digit_size,
                 N,
                 transform=None,
                 use_fixed_dataset=False,
                 random_state=2
                 )

print("the length of the test set is %d"%len(mm))
test_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=False, num_workers=0)


def train_on_batch(input_tensor, target_tensor, mask, actions, state, encoder, encoder_optimizer, criterion,teacher_forcing_ratio):
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    # input_length  = input_tensor.size(1)
    # target_length = target_tensor.size(1)
    loss = 0
    # for ei in range(input_length-1):
    #     encoder_output, encoder_hidden, output_image,_,_ = encoder(input_tensor[:,ei,:,:,:], (ei==0) )
    #     loss += criterion(output_image,input_tensor[:,ei+1,:,:,:])
    #
    # decoder_input = input_tensor[:,-1,:,:,:] # first decoder input = last image of input sequence
    #
    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # for di in range(target_length):
    #     decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input)
    #     target = target_tensor[:,di,:,:,:]
    #     loss += criterion(output_image,target)
    #     if use_teacher_forcing:
    #         decoder_input = target # Teacher forcing
    #     else:
    #         decoder_input = output_image

    pred = encoder.forward(input_tensor, actions, state, mask)
    pred = torch.stack(pred, dim=1)
    mask = mask[0, :]
    loss_function = nn.MSELoss()
    loss = loss_function(pred[:, mask[1:] == 1, ...], target_tensor[:, mask[1:] == 1, ...])

    return loss.item()



def evaluate(encoder,loader):

    with torch.no_grad():
        loss_epoch = 0
        for i, out in enumerate(loader, 0):
            idx, mask, images_input, actions, state, images_true = out
            input_tensor = images_input.to(device)
            target_tensor = images_true.to(device)
            mask = mask.to(device)
            #actions = actions.to(device)
            #state = state.to(device)


            pred = encoder.forward(input_tensor, actions, state, mask)
            pred = torch.stack(pred, dim=1)
            mask = mask[0, :]
            loss_function = nn.MSELoss()
            loss = loss_function(pred[:, mask[1:] == 1, ...], target_tensor[:, mask[1:] == 1, ...])
            loss_epoch += loss

    loss_epoch = loss_epoch / len(loader)

    print('validation mse is %.4f'%loss_epoch)

    # plot
    target_tensor = target_tensor.cpu().detach().numpy().squeeze(0).squeeze(1)  # (T-1)xHxW
    pred = pred.cpu().detach().numpy().squeeze(0).squeeze(1)
    plot_spatio_temporal_data(target_tensor, save_fig=True, fig_name='true', mask=mask[1:])
    plot_spatio_temporal_data(pred, save_fig=True, fig_name='pred', mask=mask[1:])

    return loss_epoch







phycell  =  PhyCell(input_shape=(16,16), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device)
convcell =  ConvLSTM(input_shape=(16,16), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)
encoder_rnn  = EncoderRNN(phycell, convcell, device)
encoder = EncoderDecoderRNN(encoder_rnn)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('phycell ' , count_parameters(phycell) )
print('convcell ' , count_parameters(convcell) )
print('encoder ' , count_parameters(encoder) )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder.load_state_dict(torch.load('encoder_phydnet.pth',map_location=torch.device('cpu') ))

# encoder.encoder_rnn.load_state_dict(torch.load('encoder_phydnet.pth',map_location=torch.device('cpu') ))  # use the previous training

encoder.eval()
#mse, mae,ssim = evaluate(encoder,test_loader)
mse = evaluate(encoder,test_loader)
print('mse is %.4f'%mse)

