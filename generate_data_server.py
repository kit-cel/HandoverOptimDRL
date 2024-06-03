import scipy.io
import os
import torch 
import numpy as np

this_path = os.path.dirname(os.path.abspath(__file__))

def generate_data(new_data, user, device, train_or_test, index, name_dataset): 
    ### import data from matlab ###
    mat = scipy.io.loadmat(os.path.join(this_path, 'data_from_vienna', name_dataset, train_or_test, 'rsrp_'+train_or_test+'_'+str(index)+'.mat')) 
    mat2 = scipy.io.loadmat(os.path.join(this_path, 'data_from_vienna', name_dataset, train_or_test, 'widebandSinrAllUsersdB_'+train_or_test+'_'+str(index)+'.mat'))

    rsrp_mat = mat['receivePowerdB']
    widebandsinr_mat = mat2['widebandSinrAllUsersdB']
    amount_basestations = rsrp_mat.shape[0]
    
    rsrp = rsrp_mat[:,user,:].squeeze()
    rsrp = comp_rsrq(rsrp) 
    widebandsinr = widebandsinr_mat[:,user,:].squeeze()
    time_steps = rsrp.shape[1]                                              # simulation timesteps
    
    rsrp_up = scipy.signal.resample(rsrp, time_steps*12, axis=1)            # *12 because conversion from 0,120s (MATLAB) to 0,01s
    widebandsinr_up = scipy.signal.resample(widebandsinr, time_steps*12, axis=1)

    # add moving average filter
    kernel_size = 200
    kernel = np.expand_dims(np.ones((kernel_size)) / kernel_size, axis=0)
    rsrp_filt = scipy.signal.convolve2d(rsrp_up, kernel, mode='same', boundary='symm')
    widebandsinr_filt = scipy.signal.convolve2d(widebandsinr_up, kernel, mode='same', boundary='symm')
    time_steps_filt = rsrp_filt.shape[1]
    return widebandsinr_filt, rsrp_filt, time_steps_filt, amount_basestations

def comp_rsrq(input_dbm):
    input = dBm2mW(input_dbm)
    output = np.zeros(input.shape)
    for i in range(input.shape[0]):
        sum_matrix = np.zeros(input.shape[1])
        for j in range(input.shape[0]): 
            if i != j:
                sum_matrix += input[j,:]
        output[i,:] = input[i,:] / sum_matrix
    return mW2dBm(output)


# Function to convert from mW to dBm
def mW2dBm(mW):
    return 10*np.log10(mW)

# Function to convert from dBm to mW
def dBm2mW(dBm):
    return np.power(10, dBm/10)