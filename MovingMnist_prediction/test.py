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
from models import ConvLSTM,PhyCell, EncoderRNN
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



mm = MovingMNIST(root=args.root, is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])


print("the length of the test set is %d"%len(mm))
test_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=False, num_workers=0)

constraints = torch.zeros((49,7,7)).to(device)
ind = 0
for i in range(0,7):
    for j in range(0,7):
        constraints[ind,i,j] = 1
        ind +=1



def evaluate(encoder,loader):
    total_mse, total_mae,total_ssim,total_bce = 0,0,0,0
    t0 = time.time()
    with torch.no_grad():
        for i, out in enumerate(loader, 0):
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            for ei in range(input_length-1):
                encoder_output, encoder_hidden, _,_,_  = encoder(input_tensor[:,ei,:,:,:], (ei==0))

            decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            true = torch.cat([input_tensor, target_tensor], dim=1)
            true = true.cpu().numpy()
            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions =  np.stack(predictions) # (10, batch_size, 1, 64, 64)
            predictions = predictions.swapaxes(0,1)  # (batch_size,10, 1, 64, 64)

            #mse_batch = np.mean((predictions-target)**2 , axis=(0,1,2)).sum()
            #mae_batch = np.mean(np.abs(predictions-target) ,  axis=(0,1,2)).sum()
            mse_batch = np.mean((predictions-target)**2)
            mae_batch = np.mean(np.abs(predictions-target))
            total_mse += mse_batch
            total_mae += mae_batch

            # for a in range(0,target.shape[0]):
            #     for b in range(0,target.shape[1]):
            #         total_ssim += ssim(target[a,b,0,], predictions[a,b,0,]) / (target.shape[0]*target.shape[1])
            #
            #
            # cross_entropy = -target*np.log(predictions) - (1-target) * np.log(1-predictions)
            # cross_entropy = cross_entropy.sum()
            # cross_entropy = cross_entropy / (args.batch_size*target_length)
            # total_bce +=  cross_entropy

            if i == 4:
                break
    plot_spatio_temporal_data(predictions.squeeze(0).squeeze(1), save_fig=True, fig_name='physics_net_pred')

    plot_spatio_temporal_data(true.squeeze(0).squeeze(1), save_fig=True, fig_name='physics_net_true', mask=[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])

    # print('eval mse ', total_mse/len(loader),  ' eval mae ', total_mae/len(loader),' eval ssim ',total_ssim/len(loader), ' time= ', time.time()-t0)
    print('eval mse ', total_mse / (i+1),  ' eval mae ', total_mae / (i+1), ' time= ', time.time()-t0)
    # return total_mse/len(loader),  total_mae/len(loader), total_ssim/len(loader)
    return total_mse / (i+1),  total_mae / (i+1)


phycell  =  PhyCell(input_shape=(16,16), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device)
convcell =  ConvLSTM(input_shape=(16,16), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)
encoder  = EncoderRNN(phycell, convcell, device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('phycell ' , count_parameters(phycell) )
print('convcell ' , count_parameters(convcell) )
print('encoder ' , count_parameters(encoder) )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder.load_state_dict(torch.load('encoder_phydnet.pth',map_location=torch.device('cpu') ))
encoder.eval()
#mse, mae,ssim = evaluate(encoder,test_loader)
mse, mae = evaluate(encoder,test_loader)
print('mse is %.4f'%mse)

