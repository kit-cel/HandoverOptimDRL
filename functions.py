import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import generate_data_server as gen

plotpgf = 0
if plotpgf:
    import matplotlib 
    matplotlib.use('pgf')
    plt.rcParams.update({
        "font.family": "serif",     
        "text.usetex": True,        
        "pgf.rcfonts": False,       
        'font.size' : 15
        })
    
def max(matrix, axis, i):
    val = np.amax(matrix, axis)[i]
    index = np.argmax(matrix, axis)[i]
    return val, index 

# plot reward/state of the last episode
def plot_last(array, name, this_path, sweep_name, insl_num, run_name, index_test):
    plt.figure(figsize=(11,7))
    plt.plot(array, '^')
    plt.xlabel('duration')
    plt.ylabel(name)
    plt.title(name+' last round')
    plt.savefig(os.path.join(this_path, 'results', sweep_name, insl_num, run_name, str(index_test), name +'.png'))
    plt.close()

# plot rsrp or sinr of all BS
def plot_measurement(matrix, name, this_path, sweep_name, insl_num, run_name, env, index_test, reward, state_index, plot_best_bs = 0, index = None):
    fig = plt.figure(figsize=(15,7))
    for i in range(np.shape(matrix)[0]):
        plt.plot(matrix[i,:], label='BS'+str(i))

    if plot_best_bs == 1:
        array_best_bs = []
        for j in range(np.shape(matrix)[1]-1):
            if np.isnan(index[j]) != 1:
                array_best_bs.append(matrix[index[j], j])
            else:
                array_best_bs.append(float('nan'))
        plt.plot(array_best_bs, c='b', marker='o', markersize=2, linestyle='None')#linewidth=5.0)
        plt.plot(env.hof_occur, c='r', marker='o', markersize=2, linestyle='None')
        plt.plot(env.handover_occur, c='k', marker='o', markersize=2, linestyle='None')
        np.save(os.path.join(this_path, 'results', sweep_name, insl_num, run_name, str(index_test), 'all_bs.npy'), matrix)
        np.save(os.path.join(this_path, 'results', sweep_name, insl_num, run_name, str(index_test), 'best_bs.npy'), array_best_bs)
        np.save(os.path.join(this_path, 'results', sweep_name, insl_num, run_name, str(index_test), 'hof_occur.npy'), env.hof_occur)
        np.save(os.path.join(this_path, 'results', sweep_name, insl_num, run_name, str(index_test), 'handover_occur.npy'), env.handover_occur)
        np.save(os.path.join(this_path, 'results', sweep_name, insl_num, run_name, str(index_test), 'state.npy'), index)
        np.save(os.path.join(this_path, 'results', sweep_name, insl_num, run_name, str(index_test), 'reward.npy'), reward)

    plt.legend(loc='upper right') 
    plt.xlabel('duration')
    plt.ylabel(name)    
    plt.title('measurement of '+name)
    
    if plotpgf:
        plt.savefig(os.path.join(this_path, 'results', sweep_name, insl_num, run_name, str(index_test), 'measurement'+name+'.pgf'), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(this_path, 'results', sweep_name, insl_num, run_name, str(index_test), 'measurement'+name+'.png'), bbox_inches='tight')
    plt.close()

def norm(input, clip_h, clip_l):
    input = np.clip(input, clip_l, clip_h)
    input = (input - clip_l)/(clip_h - clip_l)
    return input

def plot_loss(this_path, loss, sweep_name, insl_num, run_name, index_test):
    plt.figure(figsize=(15,7))
    plt.plot(loss)
    plt.savefig(os.path.join(this_path, 'results', sweep_name, insl_num, run_name, str(index_test), 'loss_results.png'), bbox_inches='tight')
    plt.close()

# Output: 
# 1: input < qout
# 2: qout  < input < qin 
# 3: input > qin 
def rlf(input, qin, qout):
    if (input < qout):
        return 1
    if (input < qin):
        return 2
    else:
        return 3 

def testing(env, model, user, device, this_path, insl_num, sweep_name, run_name, run_index, clip_h, clip_l, name_dataset):
    env.test = 1
    train_or_test = 'test'
    widebandsinr, rsrp, time_steps, amount_bs = gen.generate_data(0, user, device, train_or_test, run_index, name_dataset)
    # insert testing data
    env.widebandsinr = widebandsinr
    env.rsrp = norm(rsrp, clip_h, clip_l)
    env.time_steps = time_steps
    env.amount_bs = amount_bs

    obs, _ = env.reset()
    state_index_arr = []
    reward_arr = []
    info_arr = []
    for step in range(time_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        state_index_arr.append(env.state_index)
        reward_arr.append(reward)
        info_arr.append(info["mean_sinr_test"])

        # if HO execution 
        if env.HO_exe_flag == 1: 
            state_index_arr.extend(env.HO_nan_states)
            reward_arr.extend(env.HO_nan_states)
            info_arr.extend(env.HO_nan_states)
        # if RLF recovery
        if done:
            env.T310_counter = env.T310
            env.t_rlf_recovery_counter = env.t_rlf_recovery
            # also reset HO values
            env.when_HO_exe_over = env.HO_execution
            env.when_HO_prep_over = env.HO_prep
            env.HO_prep_flag = 0

            if env.terminated_by_pp == 0:
                env.hof_occur[env.t-env.t_rlf_recovery:env.t] = -45
                state_index_arr.extend(env.rlf_recov_nan)
                reward_arr.extend(env.rlf_recov_nan)
                info_arr.extend(env.rlf_recov_nan)
        if truncated:
            if len(env.state_history) - len(state_index_arr) == env.t_rlf_recovery:
                state_index_arr.extend(env.rlf_recov_nan)
                reward_arr.extend(env.rlf_recov_nan)
                info_arr.extend(env.rlf_recov_nan)
            if len(env.state_history) - len(state_index_arr) == env.HO_execution:
                state_index_arr.extend(env.HO_nan_states)
                reward_arr.extend(env.HO_nan_states)
                info_arr.extend(env.HO_nan_states)
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            plot_measurement(rsrp, 'rsrp', this_path, sweep_name, insl_num, run_name, env, run_index, reward_arr, state_index_arr, 1, state_index_arr)
            plot_last(reward_arr, 'reward', this_path, sweep_name, insl_num, run_name, run_index)
            plot_last(state_index_arr, 'state', this_path, sweep_name, insl_num, run_name, run_index)
            print("Sequence over!", "reward=", reward)
            return np.nansum(reward_arr)/len(reward_arr)