import numpy as np
import gymnasium as gym
from gymnasium import spaces
import functions as func

class NetworkEnv(gym.Env):
    def __init__(self, rsrp_list, amount_bs, device, HO_execution, HO_prep, widebandsinr_list, qin, qout, T310, t_rlf_recovery, time_steps, CONSTANT, mts, n_train_runs, n_obs):
        super(NetworkEnv, self).__init__()
        self.initial_index = 0
        self.device = device
        self.amount_bs = amount_bs
        self.state_history = [] 
        # visualize when we have a HO & HOF process 
        self.handover_occur    = np.zeros(time_steps)   
        self.handover_occur[:] = float('nan')
        self.hof_occur         = np.zeros(time_steps)
        self.hof_occur[:]      = float('nan')
        # timestep of last HO
        self.when_last_HO = None
        # HO preparation time 
        self.HO_prep = HO_prep 
        self.when_HO_prep_over = self.HO_prep
        # HO execution time
        self.HO_execution = HO_execution 
        self.when_HO_exe_over = self.HO_execution  
        # next state index 
        self.next_state_index = None
        # set time value
        self.t = 0
        # variables for RLF
        self.qin = qin
        self.qout = qout
        # rsrp for reward 
        self.rsrp_list = rsrp_list
        self.rsrp = self.rsrp_list[0]
        # CONSTANT for reward
        self.constant = CONSTANT
        # mts for PP
        self.mts = mts
        # wideband SINR for RLF reward
        self.widebandsinr_list = widebandsinr_list
        self.widebandsinr = self.widebandsinr_list[0]    # can be either train or test data
        _, self.initial_index = func.max(self.rsrp, 0, self.t)
        self.state_index = self.initial_index.item()
        # input 1 of NN: decision_bs
        # decision_bs = [0, 0, 1] -> connected to BS 2
        decision_bs = np.zeros(amount_bs)
        decision_bs[self.initial_index.item()] = 1
        # input 2 of NN: rsrp at timestep t
        self.input_rsrp = self.rsrp[:,self.t]
        # input 3 of NN: mts over or not? 0 -> over, 1 -> not over
        self.input_conn = np.array([0]) 
        # actions we can take: action_space
        self.action_space = spaces.Discrete(amount_bs)
        # RSRP is normalized between 0 and 1 
        self.low = np.zeros(n_obs)
        self.high = np.ones(n_obs)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        # set start value
        self.state = np.concatenate((decision_bs, self.input_rsrp, self.input_conn))
        self.old_state_index = None
        # test flag
        self.test = 0
        # parameteres for RLF
        self.t_rlf_recovery = t_rlf_recovery
        self.t_rlf_recovery_counter = self.t_rlf_recovery
        self.T310 = T310
        self.T310_counter = self.T310 
        self.flag_T310 = 0
        self.number_hof = 0
        # parameter for evaluation
        self.mean_sinr_test = 0
        self.time_steps = time_steps
        # for debug
        self.counter_random = 0
        # number of train files
        self.n_train_runs = n_train_runs
        # HO exe flag
        self.HO_exe_flag = 0
        # terminated by PP flag
        self.terminated_by_pp = 0
        # nan states for plotting
        self.rlf_recov_nan = np.empty(self.t_rlf_recovery)
        self.rlf_recov_nan.fill(np.nan)
        self.HO_nan_states = np.empty(self.HO_execution)
        self.HO_nan_states.fill(np.nan)
        self.pp_flag = 0 # 0 -> no PP, 1 -> PP

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        # reset pp flag
        self.pp_flag = 0
        terminated = False
        self.terminated_by_pp = False
        truncated = False
        self.state_history.append(self.state_index)
        reward = 0
        self.old_state_index = self.state_index 
        
        _, index_max = func.max(self.rsrp, 0, self.t)        
        # Apply action - no HO 
        # HO exe flag
        self.HO_exe_flag = 0
        if self.state_index == action:
            self.when_HO_prep_over = self.HO_prep 
            if func.rlf(self.widebandsinr[self.state_index, self.t], self.qin, self.qout) == 1:
                reward = -self.constant
            else:
                if self.state_index == index_max:
                    reward = self.rsrp[action, self.t] + self.constant
                else:
                    reward = self.rsrp[action, self.t]
        else:
            # save pursued next state! cannot be changed during HO exe time
            if self.when_HO_prep_over == self.HO_prep:
                self.next_state_index = action 

            if self.when_HO_prep_over == 0:
                # trigger HO execution
                self.handover_occur[self.t] = -50
                self.state_index = self.next_state_index
                self.state_history.extend(self.HO_nan_states)
                # reset values
                self.when_HO_prep_over = self.HO_prep
                # HO exe flag
                self.HO_exe_flag = 1
                if (self.t + self.HO_execution) < self.time_steps:
                    index_end = self.t + self.HO_execution
                else:
                    index_end = self.time_steps -1
                    truncated = True
                if func.rlf(self.widebandsinr[self.next_state_index, index_end], self.qin, self.qout) == 1:
                    # HOF occurs - new SINR is too bad
                    reward = - 2*self.constant
                    self.hof_occur[index_end] = -30
                    terminated = True
                    if self.test == 1:
                        self.number_hof += 1
                else:
                    # very good reward if HO to best BS, encourage agent to do HO
                    if func.rlf(self.widebandsinr[self.state_index, index_end], self.qin, self.qout) == 3:
                        reward = self.rsrp[self.next_state_index, index_end] + 2*self.constant
                    else:
                        reward = self.rsrp[self.next_state_index, index_end] + self.constant
                # Detection of PP with MTS
                if self.when_last_HO != None:
                    if (self.t - self.when_last_HO) < self.mts:
                        if self.state_history[self.when_last_HO-1] == self.next_state_index:
                            reward = - self.constant
                            # terminate training if PP discovered
                            self.pp_flag = 1
                            if terminated == False:                # set terminated if PP occurs
                                self.terminated_by_pp = True
                            terminated = True
                self.when_last_HO = self.t 
            else:
                # Continue HO preparation - We are still connected 
                # Reset all HO prep. counters if agent wants another BS during prep. 
                if self.next_state_index != action:
                    self.when_HO_prep_over = self.HO_prep
                    # new target!
                    self.next_state_index = action 

                self.when_HO_prep_over += -1 
                if func.rlf(self.widebandsinr[self.state_index, self.t], self.qin, self.qout) == 1:
                        reward = -self.constant
                else:
                    if self.state_index == index_max:
                        reward = self.rsrp[action, self.t] + self.constant
                    else:
                        reward = self.rsrp[action, self.t]


        if self.T310_counter != self.T310:
            if self.T310_counter == 0:
                # RLF/HOF
                terminated = True
                reward = - 2*self.constant
                self.hof_occur[self.t] = -40
                if self.test == 1:
                        self.number_hof += 1 
            else:
                # HOF occurs - HO execution starts but T310 runs
                if self.when_HO_prep_over == 0:
                    reward = - 2*self.constant
                    self.hof_occur[self.t] = -35
                    terminated = True
                    if self.test == 1:
                        self.number_hof += 1
                else:        
                    if func.rlf(self.widebandsinr[self.state_index, self.t], self.qin, self.qout) == 3:
                        self.T310_counter = self.T310
                    else:
                        self.T310_counter += -1 
        else: 
            if self.when_last_HO != self.t:
                if func.rlf(self.widebandsinr[self.state_index, self.t], self.qin, self.qout) == 1:
                    self.T310_counter += -1
                    reward = - self.constant

        self.mean_sinr_test += 10 ** (self.widebandsinr[self.state_index, self.t]/10)

        if self.when_last_HO != None:
            if (self.t - self.when_last_HO) < self.mts:
                self.input_conn = np.array([1])
            else:
                self.input_conn = np.array([0]) 

        decision_bs = np.zeros(self.amount_bs)
        if not np.isnan(self.state_index):
            decision_bs[self.state_index] = 1
        self.input_rsrp = self.rsrp[:,self.t+1] 
        self.state = np.concatenate((decision_bs, self.input_rsrp, self.input_conn))  
        

        if self.when_last_HO == self.t and (self.t + self.t_rlf_recovery < self.time_steps-1):
            self.t += self.HO_execution
            # add states for history if RLF recovery
        if terminated == True and (self.t + self.t_rlf_recovery < self.time_steps-1) and self.terminated_by_pp == 0:
            self.state_history.extend(self.rlf_recov_nan)
            self.t += self.t_rlf_recovery
        else:
            # abort training/eval because end of dataset reached
            if terminated == True and (self.t + self.t_rlf_recovery > self.time_steps-1) and self.terminated_by_pp == 0:
                truncated = True
        self.t += 1
        if self.t >= self.time_steps-1:
            return (np.array(tuple(self.state), dtype=np.float32), reward, False, True, {"mean_sinr_test": self.mean_sinr_test}) 
        else:
            return (np.array(tuple(self.state), dtype=np.float32), reward, terminated, truncated, {"mean_sinr_test": self.mean_sinr_test}) 
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        print("called reset")
        self.counter_random = 0
        amount_bs = self.amount_bs
        self.t = 0
        # swap the BS indices
        if self.test == 0:
            index_train_data = np.random.randint(self.n_train_runs)
            self.rsrp = self.rsrp_list[index_train_data]
            self.widebandsinr = self.widebandsinr_list[index_train_data]
            perm_order = np.random.permutation(np.shape(self.widebandsinr)[0])
            self.widebandsinr = self.widebandsinr[perm_order,:]
            self.rsrp = self.rsrp[perm_order,:]
        _, self.initial_index = func.max(self.rsrp, 0, self.t)
        self.when_HO_prep_over = self.HO_prep  
        self.state_history = []
        self.handover_occur[:] = float('nan')
        self.hof_occur[:] = float('nan')
        self.next_state_index = None
        self.state_index = self.initial_index.item()
        self.when_last_HO = None   
        self.old_state_index = None 
        self.decision_bs = np.zeros(amount_bs)
        self.decision_bs[self.initial_index] = 1
        self.input_rsrp = self.rsrp[:,self.t]
        # input 3 of NN: connected or not connected?
        self.input_conn = np.array([0])
        self.state = np.concatenate((self.decision_bs, self.input_rsrp, self.input_conn))
        self.T310_counter = self.T310 
        self.number_hof = 0
        # parameter for evaluation
        self.mean_sinr_test = 0
        # HO exe flag
        self.HO_exe_flag = 0
        # PP flag
        self.pp_flag = 0
        # terminated by PP flag
        self.terminated_by_pp = 0
        return np.array(self.state, dtype=np.float32), {} 
    
    def render(self):
        pass

    def close(self):
        pass