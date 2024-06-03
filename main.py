import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import network_env as envi
from stable_baselines3 import PPO
import os 
import glob
import functions as func
import generate_data_server as gen
import torch 
from rl_zoo3 import linear_schedule
import tensorflow as tf
import torch
import wandb

wandb.login()
### CONFIG FOLDER ###
LR = 5e-6
num_episodes = 300
name_dataset = 'ENTER_FILE_NAME_FOR_DATASET'
name_sweep = 'ENTER_FILE_NAME_FOR_SWEEP'
insl_num = '1'
### CONFIG FOLDER ###
sweep_config = {
    'name': name_sweep,
    'method': 'bayes',
    'metric': {'goal':'maximize', 'name': 'reward_sum_avg'},
    'parameters': {
        'ent_coef': {'values':[0.1, 0.01, 0.001]},
        'constant': {'max':0.95, 'min':0.6},
    }
}

sweep_name = sweep_config.get('name')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        run_name = wandb.run.name

        # start original script
        this_path = os.path.dirname(os.path.abspath(__file__))
        n_test_files = len(glob.glob(os.path.join(this_path, 'data_from_vienna', name_dataset, 'test/*.mat')))
        n_test_runs = int(n_test_files/2)
        n_train_files = len(glob.glob(os.path.join(this_path, 'data_from_vienna', name_dataset, 'train/*.mat')))
        n_train_runs = int(n_train_files/2)

        # create folder section
        if not os.path.exists(os.path.join(this_path,'results',sweep_name)):
            os.mkdir(os.path.join(this_path,'results',sweep_name))
        if not os.path.exists(os.path.join(this_path,'results',sweep_name, insl_num)):
            os.mkdir(os.path.join(this_path,'results',sweep_name, insl_num))
        if not os.path.exists(os.path.join(this_path,'results',sweep_name, insl_num, run_name)):
            os.mkdir(os.path.join(this_path,'results',sweep_name, insl_num, run_name))
        if not os.path.exists(os.path.join(this_path,'results',sweep_name, insl_num, run_name, 'tensorboard')):
            os.mkdir(os.path.join(this_path,'results',sweep_name, insl_num, run_name, 'tensorboard'))
        if not os.path.exists(os.path.join(this_path,'results',sweep_name, insl_num, run_name, 'train')): # folder to store the last epoch of the train run
            os.mkdir(os.path.join(this_path,'results',sweep_name, insl_num, run_name, 'train'))
            for i in range(n_test_runs):
                if not os.path.exists(os.path.join(this_path,'results',sweep_name, insl_num, run_name, str(i))):
                    os.mkdir(os.path.join(this_path,'results',sweep_name, insl_num, run_name, str(i)))


        # import data - load all train data in one step into environment
        train_or_test = 'train'
        user = 0
        clip_h = 10                                     # clipping of RSRP values 
        clip_l = -10
        widebandsinr_list = []
        rsrp_norm_list = []
        rsrp_list = []
        for j in range(n_train_runs):
            widebandsinr, rsrp, time_steps, amount_bs = gen.generate_data(0, user, device, train_or_test, j, name_dataset)
            rsrp_norm = func.norm(rsrp, clip_h, clip_l)
            rsrp_list.append(rsrp)
            widebandsinr_list.append(widebandsinr)
            rsrp_norm_list.append(rsrp_norm)

        # variable section
        ent_coef = config.ent_coef                      # entropy regulation
        n_batch_size = int(0.03125*time_steps)
        CONSTANT = config.constant                      # for reward/penalty
        HO_execution = 4                                # HO execution time
        HO_prep = 5                                     # HO preparation time 
        mts = 100                                       # minimum-time-of-stay 1 second
        qout = -8  
        qin = -6  
        t_rlf_recovery = 20                             # time for RLF recovery: 200 ms
        T310 = 100                                      # default value from 3GPP: 1000 ms
        n_obs = 2*amount_bs + 1                         # +1 because of third concatenate part of input array 
        
        # Validate environment
        env = envi.NetworkEnv(rsrp_norm_list, amount_bs, device, HO_execution, HO_prep, widebandsinr_list, qin, qout, T310, t_rlf_recovery, time_steps, CONSTANT, mts, n_train_runs, n_obs)
        check_env(env, warn=True)

        # NN policy
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                            net_arch=dict(pi=[64, 128, 64], vf=[64, 128, 64]))
        # Train the agent
        tensorboard_log = os.path.join(this_path,'results',sweep_name, insl_num, run_name, 'tensorboard')
        
        # load old model #                                                  
        # model = PPO.load(os.path.join(this_path, 'results', 'INSERT_PATH', 'train', 'model'), env=env, tensorboard_log=tensorboard_log, learning_rate = linear_schedule(LR), n_steps = time_steps, batch_size=n_batch_size, n_epochs = num_episodes)

        # train with new model from scratch
        model = PPO("MlpPolicy", env, ent_coef=ent_coef, learning_rate = linear_schedule(LR), verbose=1, policy_kwargs=policy_kwargs, n_steps = time_steps, batch_size=n_batch_size,  n_epochs = num_episodes, tensorboard_log=tensorboard_log).learn(time_steps*num_episodes)        
        model.save(os.path.join(this_path,'results',sweep_name, insl_num, run_name, 'train', 'model'))
        
        
        ### TESTING ##
        reward_sum_avg = 0 
        for i in range(n_test_runs):
            reward_sum_avg += func.testing(env, model, user, device, this_path, insl_num, sweep_name, run_name, i, clip_h, clip_l, name_dataset)
            print(reward_sum_avg)

        reward_sum_avg = reward_sum_avg/n_test_runs
        wandb.finish()

sweep_id = wandb.sweep(sweep=sweep_config, project=name_sweep)
wandb.agent(sweep_id, main, count=1)
