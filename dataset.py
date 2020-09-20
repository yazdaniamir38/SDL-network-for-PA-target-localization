import torch.utils.data as data
import torch
import h5py, cv2
import numpy as np
import random
import os

# train_set = DatasetFromHdf5(opt.traindata)
# training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
# os.walk("path")

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path,yes=0):
        super(DatasetFromHdf5, self).__init__()
        self.files = []
        for roots,dirs,files in file_path:
            for name in files:
                self.files.append(os.path.join(roots,name))
        self.nsamples_file = 1000
        self.yes=yes

    def __getitem__(self, index):
        file_index =  index//self.nsamples_file
        sample_index = index - file_index*self.nsamples_file
        h5 = h5py.File(self.files[file_index], 'r')
        input_image = h5['INPUT'][sample_index,:,:,:]
        target_image = h5["TARGET"][sample_index, :, :]
        if self.yes:
            clean_image=h5["Clean"][sample_index, :, :, :]
            return torch.from_numpy(input_image).float(), torch.from_numpy(clean_image).float(),torch.from_numpy(target_image).float()
        else:
            return torch.from_numpy(input_image).float(),torch.from_numpy(target_image).float()

        
    def __len__(self):
        return self.nsamples_file*len(self.files) # the total number of samples should be returned here

