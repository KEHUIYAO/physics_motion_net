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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
parser.add_argument('--nepochs', type=int, default=1, help='nb of epochs')
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=1, help='')
parser.add_argument('--save_name', type=str, default='phydnet', help='')
args = parser.parse_args()


# mm = MovingMNIST(root=args.root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2])
# train_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=True, num_workers=0)
# mm = MovingMNIST(root=args.root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2])
# validation_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=True, num_workers=0)
#
root = './'
n_frames = 20
num_digits = 2
image_size = 64
digit_size = 28
N = 4 # total number of samples including training and validation data
mask = np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,0])
mm = MovingMNIST(root,
                     n_frames,
                     mask,
                     num_digits,
                     image_size,
                     digit_size,
                     N,
                     transform=None,
                     use_fixed_dataset=False,
                     )
train_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=True, num_workers=0)

mm = MovingMNIST(root,
                 n_frames,
                 mask,
                 num_digits,
                 image_size,
                 digit_size,
                 N,
                 transform=None,
                 use_fixed_dataset=False,
                 )
validation_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=True, num_workers=0)




constraints = torch.zeros((49,7,7)).to(device)
ind = 0
for i in range(0,7):
    for j in range(0,7):
        constraints[ind,i,j] = 1
        ind +=1    

def train_on_batch(input_tensor, target_tensor, mask, actions, state, encoder, encoder_optimizer, criterion,teacher_forcing_ratio):
    encoder_optimizer.zero_grad()
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
    newloss = 0
    # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7,7]).to(device)
    for b in range(0, encoder.encoder_rnn.phycell.cell_list[0].input_dim):
        filters = encoder.encoder_rnn.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)
        m = k2m(filters.double()) 
        m  = m.float()   
        newloss += criterion(m, constraints) # constrains is a precomputed matrix
    loss = loss + newloss
    loss.backward()
    encoder_optimizer.step()
    loss = loss - newloss
    return loss.item()


def trainIters(encoder, nepochs, print_every=10,eval_every=10,name=''):
    train_losses = []
    best_mse = float('inf')

    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=0.001)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=2,factor=0.1,verbose=True)
    criterion = nn.MSELoss()
    
    for epoch in range(0, nepochs):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003) 
        
        for i, out in enumerate(train_loader, 0):
            idx, mask, images_input, actions, state, images_true = out
            input_tensor = images_input.to(device)
            target_tensor = images_true.to(device)
            mask = mask.to(device)
            #actions = actions.to(device)
            #state = state.to(device)

            loss = train_on_batch(input_tensor, target_tensor, mask, actions, state, encoder, encoder_optimizer, criterion, teacher_forcing_ratio)
            loss_epoch += loss

        loss_epoch = loss_epoch / len(train_loader)
        train_losses.append(loss_epoch)

        if (epoch+1) % print_every == 0:
            #print('epoch ',epoch,  ' loss ',loss_epoch, ' time epoch ',time.time()-t0)
            print('training mse is %.4f'%loss_epoch)
        if (epoch+1) % eval_every == 0:
            #mse, mae,ssim = evaluate(encoder,validation_loader)
            mse = evaluate(encoder,validation_loader)
            scheduler_enc.step(mse)
        torch.save(encoder.state_dict(),'encoder_{}.pth'.format(name))
    return train_losses

    
def evaluate(encoder,loader):
    total_mse, total_mae,total_ssim,total_bce = 0,0,0,0
    t0 = time.time()
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


if __name__ == "__main__":
    trainIters(encoder,args.nepochs,print_every=args.print_every,eval_every=args.eval_every,name=args.save_name)


