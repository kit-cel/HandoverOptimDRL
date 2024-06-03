import numpy as np
import matplotlib.pyplot as plt
import functions as func 
import scipy.io
import os
import matplotlib 
import glob 
pgfplot = 0
downsample_factor = 1
data_set_name = '1_8GHZ_total'
plt.rcParams.update({'font.size': 10})
if pgfplot == 1:
    matplotlib.use('pgf')
    plt.rcParams.update({
        "font.family": "serif",  
        "text.usetex": True,     
        "pgf.rcfonts": False,     
       'font.size' : 30
        })

this_path = os.path.dirname(os.path.abspath(__file__))

n_test_files = len(glob.glob(os.path.join(this_path, '../data_from_vienna', data_set_name, 'test/*.mat')))
n_runs = int(n_test_files/2)

def main(index):
    this_path = os.path.dirname(os.path.abspath(__file__))
    ### import data from matlab ###
    mat = scipy.io.loadmat(os.path.join(this_path, '../data_from_vienna', data_set_name, 'test/', 'rsrp_test_'+str(index)+'.mat'))
    mat2 = scipy.io.loadmat(os.path.join(this_path, '../data_from_vienna', data_set_name, 'test/','widebandSinrAllUsersdB_test_'+str(index)+'.mat'))

    # add moving average filter
    kernel_size = int(200)
    kernel = np.expand_dims(np.ones((kernel_size)) / kernel_size, axis=0)

    matrix_ = mat['receivePowerdB']
    matrix  = np.expand_dims(scipy.signal.convolve2d(scipy.signal.resample(matrix_, matrix_.shape[2]*int(12), axis=2).squeeze(), kernel, mode='same', boundary='symm'), axis=1)
    widebandsinr_mat_ = mat2['widebandSinrAllUsersdB']
    widebandsinr_mat = np.expand_dims(scipy.signal.convolve2d(scipy.signal.resample(widebandsinr_mat_, widebandsinr_mat_.shape[2]*int(12), axis=2).squeeze(), kernel, mode='same', boundary='symm'), axis=1)
    
    amount_basestations = matrix.shape[0]
    time_steps = matrix.shape[2]                    # simulation timesteps
    user= 0
    measurement = matrix[:,user,:].squeeze()
    widebandsinr = widebandsinr_mat[:,user,:].squeeze()

    ### set variables ###
    threshold = -80
    hysteresis = 1
    offset = 2
    ttt_step = int(16)
    ttt = ttt_step                                  # time to trigger for A3
    ttt_a2_step = int(16)                           # time to trigger for A2
    ttt_a2 = ttt_a2_step
    timerstep = int(4)                              # HO Execution Time
    T304 = timerstep                
    ho_prep_time = int(5)                           # HO Preparation Time
    ho_prep = ho_prep_time                      
    flag_ttt = 0                                    # 0: ttt not triggered, 1: ttt already triggered
    flag_ttt_a2 = 0
    flag_ho_prep = 0                                # 0: no handover preparation, 1: handover preparation pending
    flag_handover = 0                               # 0: no handover, 1: handover pending
    handover_timeline    = np.empty(time_steps)     # visualize to which BS the UE is attached
    handover_timeline[:] = np.nan
    handover_occur       = np.zeros(time_steps)     # visualize if we have a handover process
    handover_occur[:] = np.nan
    sinr_timeline     = np.zeros(time_steps)
    sinr_timeline[:] = np.nan
    sinr_timeline_max = np.zeros(time_steps)

    # variables for RLF, HOF, PP
    qout = -8
    qin = -6
    mts = int(100)                                  # minimum-time-of-stay 1 second
    t310_steps = int(100)
    T310 = t310_steps
    flag_t310 = 0
    t_rlf_recovery = int(20) 
    t_rlf_recovery_counter = t_rlf_recovery

    num_hof = 0                                     # handover failure
    num_pp  = 0                                     # ping-pong

    ### start simulation ###
    # use event A2 and A3
    i = 0

    max_val, max_index = func.max(measurement, 0, i)

    current_index = max_index
    supposed_index = current_index 

    while i < time_steps:
        handover_timeline[i] = current_index

        max_val, max_index = func.max(measurement, 0, i)

        var_a2 = func.event_a2(measurement[current_index, i], threshold, hysteresis)

        if flag_handover == 1:
            var_a3 = func.event_a3(measurement[current_index, i], measurement[supposed_index, i], offset, hysteresis)
        else:
            var_a3 = func.event_a3(measurement[current_index, i], measurement[max_index, i], offset, hysteresis)
            
        rlf = func.rlf(widebandsinr[current_index, i],qin, qout)

        if t_rlf_recovery_counter == t_rlf_recovery:
            if flag_t310 == 1:
                if T310 == 0:
                    # declare HOF
                    t_rlf_recovery_counter += -1
                    num_hof += 1
                else:
                    if ho_prep == 0:
                        # declare HOF
                        t_rlf_recovery_counter += -1
                        num_hof += 1
                        flag_t310 = 0
                        T310 = t310_steps
                    else:
                        if rlf == 3:
                            # reset T310
                            T310 = t310_steps
                            flag_t310 = 0
                        else:
                            T310 += -1
            else:
                if T304 == 0:
                    if func.rlf(widebandsinr[supposed_index, i],qin, qout) == 1:
                        # declare HOF
                        t_rlf_recovery_counter += -1
                        num_hof += 1
                        flag_t310 = 0
                        T310 = t310_steps
                else:
                    if rlf == 1:
                        flag_t310 = 1
                        T310 += -1
                        sinr_timeline[i] = 10 ** (widebandsinr_mat[current_index,user,i].squeeze()/10)

            if var_a2 == 1 or (flag_handover == 1) or (var_a2 == 0 and flag_ttt_a2 == 1):
                if flag_ttt_a2 == 0:
                    if ttt_a2 == 0:
                        flag_ttt_a2 = 1
                        ttt_a2 = ttt_a2_step
                    else:
                        ttt_a2 += -1
                    sinr_timeline[i] = 10 ** (widebandsinr_mat[current_index,user,i].squeeze()/10)
                else:
                    if var_a3 == 1:
                        if flag_ttt == 0:
                            if ttt == 0:
                                flag_ttt = 1 
                                ttt = ttt_step 
                                supposed_index = max_index  # current_bs selects best bs possible
                            else:
                                ttt = ttt - 1
                            sinr_timeline[i] = 10 ** (widebandsinr_mat[current_index,user,i].squeeze()/10)

                        else:
                            if ho_prep == 0:
                                flag_ho_prep = 0
                                if T304 == timerstep:
                                    handover_occur[i] = 1
                                if T304 == 0:
                                    # handover process finished, reset all flags
                                    flag_ttt = 0
                                    ttt = ttt_step
                                    ttt_a2 = ttt_a2_step
                                    flag_ttt_a2 = 0
                                    flag_handover = 0
                                    ho_prep = ho_prep_time
                                    T304 = timerstep

                                    current_index = supposed_index
                                    handover_timeline[i] = current_index
                                    sinr_timeline[i] = 10 ** (widebandsinr_mat[current_index,user,i].squeeze()/10)
                                else:
                                    # handover process start (after TTT&HO preparation)
                                    flag_handover = 1
                                    T304 = T304 - 1
                                    sinr_timeline[i] = 0
                            else:
                                # start handover preparation
                                flag_ho_prep = 1
                                ho_prep = ho_prep - 1
                                sinr_timeline[i] = 10 ** (widebandsinr_mat[current_index,user,i].squeeze()/10)
                    else:
                        if var_a3 == -1:
                            # discard handover process
                            flag_ttt = 0
                            ttt = ttt_step
                            flag_handover = 0
                            ho_prep = ho_prep_time
                            T304 = timerstep
                        else:
                            if flag_ho_prep == 1:
                                ho_prep = ho_prep -1
                            if flag_handover == 1:
                                T304 = T304 - 1 
                        sinr_timeline[i] = 10 ** (widebandsinr_mat[current_index,user,i].squeeze()/10)
            else:
                # discard handover process
                flag_ttt = 0
                ttt_a2 = ttt_a2_step
                ttt = ttt_step
                flag_handover = 0
                T304 = timerstep
                ho_prep = ho_prep_time
                sinr_timeline[i] = 10 ** (widebandsinr_mat[current_index,user,i].squeeze()/10)
        else:
            if t_rlf_recovery_counter != 0:
                t_rlf_recovery_counter += -1
            else:
                # choose max index
                current_index = max_index
                # reset values
                flag_ttt = 0
                flag_t310 = 0
                ttt = ttt_step
                ttt_a2 = ttt_a2_step
                flag_ttt_a2 = 0 
                flag_handover = 0
                ho_prep = ho_prep_time
                T304 = timerstep
                t_rlf_recovery_counter = t_rlf_recovery

        i = i + 1

    sinr_timeline_max = (10 ** (widebandsinr/10)).max(axis=0)
    num_pp = func.calc_pp(handover_occur, handover_timeline, mts)
    rate, average_data_rate_this_run = func.calc_rate(sinr_timeline_max, sinr_timeline)

    print(handover_timeline, end='\n\n\n')
    print('******* results *******')
    print('Total number of handover failures:', num_hof)
    print('Total number of ping-pongs:', num_pp)
    print('The percentage of the achieved rate is:', rate*100,'%')

    plt.figure(figsize=(55,25))

    xnew = np.linspace(0, len(handover_occur), int(len(handover_occur)/downsample_factor), endpoint=False)
    rsrp_plot = np.empty(time_steps)
    no_rsrp_plot = np.empty(time_steps)
    no_rsrp_plot[:] = np.nan
    rsrp_plot[:] = np.nan
    for j in range(amount_basestations):
        plt.plot(xnew, measurement[j,:][::downsample_factor], label='BS'+str(j))
    for k in range(time_steps):
        if not np.isnan(sinr_timeline[k]):
            if sinr_timeline[k] == 0:
                rsrp_plot[k] = np.nan
            else:
                rsrp_plot[k] = matrix[int(handover_timeline[k]),user,k]
        else:
            no_rsrp_plot[k] = -50

    plt.plot(xnew, rsrp_plot[::downsample_factor], c='b', marker='o', markersize=2, linestyle='None', label='RSRP conn.')
    plt.plot(xnew, no_rsrp_plot[::downsample_factor], c='r', marker='s', markersize=2, linestyle='None', label='HOF')
    plt.plot(handover_occur-50, c='k',marker='X', label='HO', linestyle='None', markersize=2)
    plt.legend(loc='upper right')
    plt.ylabel('RSRP in dBm')
    plt.xlabel('timesteps in 10 ms') 
    plt.xlim((0,len(handover_occur)))
    plt.grid()
    if pgfplot == 1:
        plt.savefig(os.path.join(this_path, 'results/rsrp_3gpp_'+ str(index)+'.pdf'))
    else:
        plt.show()
    return average_data_rate_this_run

# start the whole script
average_data_rate = 0
for i in range(n_runs):
    average_data_rate += main(i)
