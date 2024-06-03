import numpy as np

# -1: cancel event
#  0: do nothing
#  1: start handover

# Serving cell becomes better than a configured threshold 
def event_a1(input, threshold, hysteresis):
    if input >= threshold + hysteresis:
        return 1
    else:
        if input <= threshold - hysteresis:
            return -1
        else:
            return 0
    
# Serving cell becomes worse than a configured threshold
def event_a2(input, threshold, hysteresis):
    if input >= threshold + hysteresis:
            return -1
    else:
        if input <= threshold - hysteresis:
            return 1
        else:
            return 0

# Neighbor cell becomes by an offset better than another given cell
def event_a3(input, neighbour, offset, hysteresis):
    if neighbour >= input + offset + hysteresis:
        return 1
    else:
        if neighbour <= input + offset - hysteresis:
            return -1
        else:
            return 0
        
# Another given cell becomes worse than a configured threshold and neighbor cell becomes better than another configured threshold
def event_a5(input, neighbour, threshold_high, threshold_low, hysteresis):
    if (input <= threshold_low - hysteresis) and (neighbour >= threshold_high + hysteresis):
        return 1
    else:
        if (input >= threshold_low + hysteresis) or (neighbour <= threshold_high - hysteresis):
            return -1
        else:
            return 0
        

def max(matrix, axis, i):
    val = np.amax(matrix, axis)[i]
    index = np.argmax(matrix, axis)[i]
    return val, index 

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
    
def calc_pp(handover_occur, handover_timeline, mts):
    num_pp = 0
    occur_index = np.where(handover_occur == 1)[0]
    if occur_index.size > 1:
        for k in range(occur_index.size - 1):
            if handover_timeline[occur_index[k]] == handover_timeline[occur_index[k + 1] + 1]:
                if (occur_index[k + 1] - occur_index[k]) < mts:
                    num_pp = num_pp + 1
    return num_pp

# calculates the percentage of the achieved rate 
def calc_rate(SINR_a, SINR_real):
    SINR_log_max = np.log2(SINR_a+1)
    SINR_real_without_zeros = SINR_real[SINR_real != 0]
    SINR_log_real = np.log2(SINR_real_without_zeros+1)

    SINR_mean_max = np.mean(SINR_log_max)
    SINR_real_mean = np.nansum(SINR_log_real)/len(SINR_real)
    return SINR_real_mean/SINR_mean_max, SINR_real_mean