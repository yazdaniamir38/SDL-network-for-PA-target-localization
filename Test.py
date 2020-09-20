#Testing a pretrained network
import torch
import numpy as np
import scipy.io as si
import h5py
import os
import Network
from Network import CoordRegressionNetwork
from Network import final
#The architecture for the trained model
mode='SDL'
if mode =='SDL'
    model=final(n_locations=4).cuda()
else:
    model=CoordRegressionNetwork(n_locations=4).cuda()
checkpoint='/model/SDL_on_mixed.pth'
model.load_state_dict(torch.load(checkpoint))
test_path='/dataset/test/'
with h5py.File(''.join([test_path, 'Codes_to_Share/X_Test_mixed.mat']), 'r') as data:
    X_Test = data['XTest'][:]
X_Test = np.transpose(X_Test)
X_Test = torch.from_numpy(X_Test)
data = si.loadmat(''.join([data_path, 'Codes_to_Share/Y_Test_mixed.mat']))
Y_te = data['YTest'][:]
Y_te = torch.from_numpy(Y_te).float()
Y_Test = torch.Tensor(6000, 4, 2)
max_ax = 50
min_ax = 10
max_normalized_depth = 0.9

Y_Test[:, :, 0] = Y_te[:,:,1]
Y_Test[:, :, 1] = Y_te[:,:,0]
bs=4
batch = 0
errors = torch.Tensor(6000, 12, 2)
heat = torch.Tensor(6000, 4, 256, 512)
nplabels = torch.Tensor(bs, 4, 2)
npcoords = torch.Tensor(bs, 4, 2)
bw = bs  # batch width to decrease at last batch
test=torch.utils.data.TensorDataset(X_Test, Y_Test)
testloader = torch.utils.data.DataLoader(test, batch_size=bs, shuffle=False)
for data in testloader:
    images, labels = data
    coords, heatmaps, _, _ = model(images.cuda().float())

    if batch * bs + bw >= errors.shape[0]:
        bw = errors.shape[0] - batch * bs
        nplabels = torch.Tensor(bw, 4, 2)
        npcoords = torch.Tensor(bw, 4, 2)

    for i in range(bw):
        print(1 + i + batch * bs)
        nplabels[i, :, :] = labels[i, :,:]
        npcoords[i, :, 0] = coords.detach().cpu()[i, :, 0] * ((max_ax - min_ax) / 2 / max_normalized_depth) + (
                    (max_ax - min_ax) / 2 + min_ax)
        npcoords[i, :, 1] = coords.detach().cpu()[i, :, 1] * ((max_ax - min_ax) / 2 / max_normalized_depth)
        print(nplabels[i])
        print(npcoords[i])
        print(bw)

    heat[batch * bs:batch * bs + bw, :, :, :] = heatmaps.detach().cpu()
    errors[batch * bs:batch * bs + bw, 0:4, :] = nplabels
    errors[batch * bs:batch * bs + bw, 4:8, :] = npcoords
    errors[batch * bs:batch * bs + bw, 8:12, :] = abs(npcoords - nplabels)
    batch += 1
axlow = np.mean(errors[:, 8:12, :].numpy()[np.where(errors[:, 0:4, 0] < 35), 0])
latlow = np.mean(errors[:, 8:12, :].numpy()[np.where(errors[:, 0:4, 0] < 35), 1])
axhigh = np.mean(errors[:, 8:12, :].numpy()[np.where(errors[:, 0:4, 0] >= 35), 0])
lathigh = np.mean(errors[:, 8:12, :].numpy()[np.where(errors[:, 0:4, 0] >= 35), 1])
euclow = np.mean(np.sqrt(np.power(errors[:, 8:12, :].numpy()[np.where(errors[:, 0:4, 0] < 35), 0], 2) + np.power(
    errors[:, 8:12, :].numpy()[np.where(errors[:, 0:4, 0] < 35), 1], 2)))
euchigh = np.mean(np.sqrt(np.power(errors[:, 8:12, :].numpy()[np.where(errors[:, 0:4, 0] >= 35), 0], 2) + np.power(
    errors[:, 8:12, :].numpy()[np.where(errors[:, 0:4, 0] >= 35), 1], 2)))
latmean = np.mean(errors[:, 8:12, 1].numpy())
latstd = np.std(errors[:, 8:12, 1].numpy())
axmean = np.mean(errors[:, 8:12, 0].numpy())
axstd = np.std(errors[:, 8:12, 0].numpy())
eucmean = np.mean(np.sqrt(np.power(errors[:, 8:12, 0].numpy(), 2) + np.power(errors[:, 8:12, 1].numpy(), 2)))
eucstd = np.std(np.sqrt(np.power(errors[:, 8:12, 0].numpy(), 2) + np.power(errors[:, 8:12, 1].numpy(), 2)))

print('\n Mean lateral error: %f' % latmean)
print('\n STD of lateral error: %f' % latstd)
print('\n Mean axial error: %f' % axmean)
print('\n STD of axial error: %f' % axstd)
print('\n Mean euclidian error: %f' % eucmean)
print('\n STD of euclidian error: %f' % eucstd)
