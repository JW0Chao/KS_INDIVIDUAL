import numpy as np
from KS import KS
import gc
import os
import torch
import matplotlib.pyplot as plt

import train
import buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

U_bf = np.loadtxt('u3.dat')  #select u1, u2 or u3 as target solution
x = np.loadtxt('x.dat')      #select space discretization of the target solution 

MAX_EPISODES = 1000         #total epoch iterations
MAX_STEPS = 3000             #total steps in one epoch
MAX_TOTAL_REWARD = -35  #critic reward to Failure 
S_DIM = 8                  #number of equispaced sensors
A_DIM = 4                    #number of equispaced actuators
A_MAX = 0.5                  #maximum amplitude for the actuation [-A_MAX, A_MAX]
L = 22
EXP_NAME = '8sensors_topshap'  # separate run for SHAP-selected top sensors
MODEL_DIR = f'./Model_{EXP_NAME}'
BUFFER_DIR = f'./Buffer_{EXP_NAME}'

ks = KS(L=L,N=x.size,a_dim=A_DIM)  #Kuramoto-Sivashinsky class initialization

if S_DIM == 8:
    sensor_indices = np.array([ 4,12,20,28,36,44,52,60], dtype=np.int64)
    if np.unique(sensor_indices).size != S_DIM:
        raise ValueError("Top-SHAP sensor list for 8-sensor setup contains duplicates.")
    if np.any(sensor_indices < 0) or np.any(sensor_indices >= x.shape[0]):
        raise ValueError(f"Top-SHAP sensor list contains out-of-bounds index for x_size={x.shape[0]}.")
else:
    sensor_step = x.shape[0] // S_DIM
    if sensor_step <= 0:
        raise ValueError(f"Invalid sensor step: x_size={x.shape[0]}, S_DIM={S_DIM}")
    sensor_indices = np.arange(0, x.shape[0], sensor_step, dtype=np.int64)
    if sensor_indices.size != S_DIM:
        raise ValueError(
            f"Sensor slicing produced {sensor_indices.size} sensors, expected {S_DIM}. "
            "Adjust S_DIM or sensor selection logic."
        )


print('State Dimensions :- ', S_DIM)
print('Action Dimensions :- ', A_DIM)
print('Action Max :- ', A_MAX)
print('Sensor Indices :- ', sensor_indices.tolist())
print('Sensor Positions :- ', x[sensor_indices].tolist())


Restart = False              #restart the optimization
Test = False                #Test mode: load and use the policy w/o optimization
Plot = False           #if true plot the target solution and the actual solution
ini = 0                      #number of epoch for restart or test the policy 

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BUFFER_DIR, exist_ok=True)
ram = buffer.MemoryBuffer(buffer_dir=BUFFER_DIR)                                         #memory class initialization
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram, device, Test, model_dir=MODEL_DIR, buffer_dir=BUFFER_DIR)     #RL class initialization 

if Test:
    init_states = np.loadtxt('INIT.dat')
    if init_states.ndim == 1:
        init_states = init_states.reshape(1, -1)
    if init_states.shape[1] != x.size:
        raise ValueError(f"INIT.dat row length ({init_states.shape[1]}) must match x size ({x.size})")

if Restart or Test:
    if Test and not os.path.exists(os.path.join(MODEL_DIR, str(ini) + '_actor.pt')):
        actor_ckpts = [f for f in os.listdir(MODEL_DIR) if f.endswith('_actor.pt') and not f.endswith('_target_actor.pt')]
        if not actor_ckpts:
            raise FileNotFoundError(f"No actor checkpoints found in {MODEL_DIR}")
        ini = max(int(f.split('_')[0]) for f in actor_ckpts)
        print('Test checkpoint not found for ini, using latest episode:', ini)
    trainer.load_models(ini,Test)                                   #load saved_model
     
for _ep in range(ini,MAX_EPISODES):
    if Test:
        init_idx = np.random.randint(init_states.shape[0])
        new_observation = np.float32(init_states[init_idx].copy())  #random test initial condition from INIT.dat
    else:
        new_observation = np.loadtxt('u2.dat')                      #load training initial condition
    for r in range(MAX_STEPS):
        state = np.float32(new_observation[sensor_indices])
        observation = new_observation
        action = trainer.get_action(state, Test=Test)
        new_observation = ks.advance(observation,action)
        reward = -np.linalg.norm(new_observation-U_bf)              #reward evaluation 
        new_state = np.float32(new_observation[sensor_indices])
        
        # push this exp in ram
        if reward < MAX_TOTAL_REWARD and not Test:
            reward = -100
            trainer.ram.add(state, action, reward, new_state, Test)
            break
        trainer.ram.add(state, action, reward, new_state, Test)

        # perform optimization
        trainer.optimize(Test)
        if r%20 == 0 and Plot:
            plt.clf()
            plt.plot(x,U_bf)
            plt.plot(x,new_observation)
            plt.plot(x[sensor_indices],new_observation[sensor_indices],'o')
            plt.pause(0.05)
            plt.show(block=False)
            
		
        
    trainer.update_pert(Test)
    gc.collect()

    print('EPISODE :- ', _ep, 'rew: ', np.float32(reward), 'memory: ', 
          np.float32(trainer.ram.len/trainer.ram.maxSize*100),'% ', 'update: ', np.float32(trainer.update),
          ' c_loss: ', np.float32(trainer.last_critic_loss), ' a_loss: ', np.float32(trainer.last_actor_loss))
    if _ep%10 == 0 and not Test:
        trainer.save_models(_ep)
        np.savetxt(os.path.join(BUFFER_DIR, 'state.dat'), observation)
        np.savetxt(os.path.join(BUFFER_DIR, 'action.dat'), ks.f0)
        np.savetxt(os.path.join(BUFFER_DIR, 'new_state.dat'), new_observation)


print('Completed episodes')
