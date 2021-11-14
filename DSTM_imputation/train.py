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
from constrain_moments import K2M
# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--nepochs', type=int, default=20, help='nb of epochs')
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=1, help='')
parser.add_argument('--save_name', type=str, default='phydnet', help='')
args = parser.parse_args()


class DatasetDstm3:
    """class for generating data from a basic dynamical spatio-temporal model
    The high values are transferring from the top left to right down
    Denote:
    Y_t: a vector containing the underlying process of all spatial locations at time t
    Z_t = Y_t + sptial_error_term

    Y_t(s) = \Sum_{x=1}^{n^2} m(s, x | \theta_1, \theta_2, \theta_3, \theta_4) * Y_{t-1}(x) + \eta_{t}, where \eta_{t} \sim Gau(0, R_{1t})
    and m(s, x | \theta_1, \theta_2, \theta_3, \theta_4) = \theta_1 * \exp( - 1 / \theta_2 * [(x1 - \theta_1 - s1)^2 + (x2 - \theta_2 -s2)^2])



    Attributes:
          Z: generated dataset
          mask: a vector of T containing 0 and 1s, indicating the missing patterns of the sequence
          baseline_underlying: Y_0(s)

    """

    def __init__(self,
                 n,
                 T,
                 theta1,
                 theta2,
                 theta3,
                 theta4,
                 total,
                 mask,
                 baseline_underlying
                 ):
        """
        :param n: spatial grid of nxn
        :param T: temporal dimension
        :param theta1: parameter in the linear dstm model
        :param theta2: parameter in the linear dstm model
        :param theta3: parameter in the linear dstm model
        :param theta4: parameter in the linear dstm model
        :param total: total number of training samples
        :param mask: a numpy vector of T containing 0 and 1s, indicating the missing patterns of the sequence
        :param baseline_underlying: a matrix or a list of matrix, representing Y_0(s) of the ith training sample


        """


        self.mask = mask

        # if baseline_underlying is a scalar, then first expand it to be a numpy array, and assuming each training sample has the same baseline
        if not hasattr(baseline_underlying, "__len__"):
            self.baseline_underlying = [np.ones(n**2) * baseline_underlying for i in range(total)]
        else:
            if baseline_underlying.ndim == 1:  # assuming each training sample has the same baseline
                self.baseline_underlying = [baseline_underlying for i in range(total)]
            else:
                self.baseline_underlying = baseline_underlying



        self.Z = self.prepare_data(n, T, theta1, theta2, theta3, theta4, total)




    def prepare_data(self, n, T, theta1, theta2, theta3, theta4, total):
        """generate data from the dstm model


        :param n: spatial grid of nxn
        :param T: temporal dimension
        :param gamma: parameter in the linear dstm model
        :param l: parameter in the linear dstm model
        :param offset: parameter in the linear dstm model
        :param total: total number of training samples
        :return: a tuple of tensors , both of size (total x T x 1 x n x n)
        """



        location_list = []  # create a list storing each spatial location
        for i in range(n):
            for j in range(n):
                location_list.append([i, j])
        location_list = np.array(location_list)

        distance_matrix_list = [
        ]  # create a list, each element stores the pairwise distance between it and every spatial location
        for i in range(n * n):
            dist = np.array([
                np.sqrt((x[0] - theta3)**2 + (x[1] - theta4)**2)
                for x in location_list[i] - location_list
            ])
            distance_matrix_list.append(dist)
        distance_matrix_list = np.array(distance_matrix_list)

        weights_matrix = theta1 * np.exp(
            -(distance_matrix_list)**2 / theta2
        )  # create a matrix, each row stores the weights matrix between it and every spatial location

        ## normalize the weights
        # def normalize(x):
        #     return x / np.sum(x)
        #
        # weights_matrix = np.apply_along_axis(normalize, 1, weights_matrix)

        # check stability of the evolving process
        w, _ = np.linalg.eig(weights_matrix)
        max_w = np.max(w)
        if max_w == 1 or max_w > 1:
            print("max eigen value is %f" % max_w)
            raise(ValueError("change initial parameters!"))
        else:
            print("max eigen value is %f" % max_w)
            print("valid initial parameters!")

        # random error terms
        eta = np.random.randn(n * n, T, total) * 0.01

        # simulate obs
        Z = np.zeros((n * n, T, total))
        Y = np.zeros((n * n, T, total))




        for i in range(total):
            Y[:, 0, i] = self.baseline_underlying[i]


            for t in range(1, T):
                Y[:, t, i] = np.dot(weights_matrix, (Y[:, (t - 1),
                                                     i])[:, None]).ravel() + eta[:, t, i]



        # normalization
        scaled_Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

        # add error term
        for i in range(total):
            for t in range(T):
                Z[:, t, i] = scaled_Y[:, t, i ] + eta[:, t, i]


        Z = Z.reshape((n, n, T, total))  # convert data to n x n x T x total
        Z = Z[..., None]
        Z = Z.transpose(3, 2, 4, 0, 1)  # convert data to total x T x 1 x n x n





        # the best we can do is that we fit everything except the spatial error terms
        print(
            "based on the error term, the best mse we can achieve will be above %.4f"
            % np.mean(eta**2))

        return Z.astype(np.float32)

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        Z = self.Z[idx]  # idx th sample

        mask = self.mask

        # the mask starts with 1, for example 110100, the input is the observed temporal snapshots aggregated together, in this case, it is [Z[0:2, :], Z[3, :]]
        # split the array Z into chuncks following the indexes in the masks where 0 and 1 alternates
        images_input = Z[mask == 1, ...]
        images_true = Z[1:, ...]
        actions = []
        state = []
        return [idx, mask, images_input, actions, state, images_true]

# simulate the data
n = 64
T = 15
theta1 = 0.5
theta2 = 1
theta3 = 1
theta4 = 1
total = 1600
mask = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]) == 1
baseline_underlying = np.random.randn(total, n**2)  # the baseline changes for every sample
data = DatasetDstm3(n, T, theta1, theta2, theta3, theta4, total, mask, baseline_underlying)

train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=args.batch_size, shuffle=False, num_workers=0)

data = DatasetDstm3(n, T, theta1, theta2, theta3, theta4, total, mask, baseline_underlying)

validation_loader =  torch.utils.data.DataLoader(dataset=data, batch_size=args.batch_size, shuffle=False, num_workers=0)



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



