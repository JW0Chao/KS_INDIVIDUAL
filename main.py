import numpy as np
from KS import KS
import gc
import os
import json
import shutil
import torch
import matplotlib.pyplot as plt

import train
import buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

U_bf = np.loadtxt('u3.dat')  #select u1, u2 or u3 as target solution
x = np.loadtxt('x.dat')      #select space discretization of the target solution 

MAX_EPISODES = 500         #total epoch iterations
MAX_STEPS = 5000             #total steps in one epoch
MAX_TOTAL_REWARD = -35  #critic reward to Failure 
S_DIM = 8                  #number of equispaced sensors
A_DIM = 4                    #number of equispaced actuators
A_MAX = 0.5                  #maximum amplitude for the actuation [-A_MAX, A_MAX]
L = 22
EXP_NAME = '8sensors_topshap'  # separate run for SHAP-selected top sensors
MODEL_DIR = f'./Model_{EXP_NAME}'
BUFFER_DIR = f'./Buffer_{EXP_NAME}'

# Validation-driven early-stop configuration.
EARLY_STOP_ENABLED = True
VAL_INIT_FILE = "INIT.dat"
VAL_SEED = 0
VAL_SIZE = 20
VAL_MAX_STEPS = 1500
VAL_FINAL_WINDOW_FRAC = 0.2
VAL_EPSILON_BETA = 0.10
VAL_DWELL_TIME = 1.0
EVAL_INTERVAL = 25
MIN_EPISODES = 200
PATIENCE_EVALS = 6
DELTA_SR = 0.05
DELTA_ERR = 0.02
BEST_SCORE_ALPHA = 0.25

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

init_states = np.loadtxt(VAL_INIT_FILE)
if init_states.ndim == 1:
    init_states = init_states.reshape(1, -1)
if init_states.shape[1] != x.size:
    raise ValueError(f"{VAL_INIT_FILE} row length ({init_states.shape[1]}) must match x size ({x.size})")

if Restart or Test:
    if Test and not os.path.exists(os.path.join(MODEL_DIR, str(ini) + '_actor.pt')):
        actor_ckpts = [f for f in os.listdir(MODEL_DIR) if f.endswith('_actor.pt') and not f.endswith('_target_actor.pt')]
        if not actor_ckpts:
            raise FileNotFoundError(f"No actor checkpoints found in {MODEL_DIR}")
        ini = max(int(f.split('_')[0]) for f in actor_ckpts)
        print('Test checkpoint not found for ini, using latest episode:', ini)
    trainer.load_models(ini,Test)                                   #load saved_model
     
current_ep = None
last_observation = None
last_new_observation = None


def first_stable_step(error_curve, epsilon, dwell_steps):
    n = error_curve.shape[0]
    if dwell_steps <= 1:
        hits = np.where(error_curve <= epsilon)[0]
        return int(hits[0]) if hits.size > 0 else None
    if dwell_steps > n:
        return None
    for t in range(0, n - dwell_steps + 1):
        if np.all(error_curve[t:t + dwell_steps] <= epsilon):
            return t
    return None


def _save_training_snapshot(ep):
    trainer.save_models(ep)
    if last_observation is not None:
        np.savetxt(os.path.join(BUFFER_DIR, 'state.dat'), last_observation)
    np.savetxt(os.path.join(BUFFER_DIR, 'action.dat'), ks.f0)
    if last_new_observation is not None:
        np.savetxt(os.path.join(BUFFER_DIR, 'new_state.dat'), last_new_observation)


def _copy_best_checkpoint_aliases(ep):
    ckpt_map = {
        f"{ep}_actor.pt": "best_actor.pt",
        f"{ep}_critic.pt": "best_critic.pt",
        f"{ep}_target_actor.pt": "best_target_actor.pt",
        f"{ep}_target_critic.pt": "best_target_critic.pt",
    }
    for src_name, dst_name in ckpt_map.items():
        src = os.path.join(MODEL_DIR, src_name)
        dst = os.path.join(MODEL_DIR, dst_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Expected checkpoint not found for best alias copy: {src}")
        shutil.copy2(src, dst)


def _append_validation_history_row(path, row):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(
            f"{row['episode']},{row['success_rate']:.6f},{row['mean_final_error']:.6f},"
            f"{row['score']:.6f},{row['sr_gain']:.6f},{row['err_rel_gain']:.6f},"
            f"{row['no_improve_eval_count']},{int(row['meaningful_improvement'])},"
            f"{row['best_score']:.6f},{row['best_score_episode']}\n"
        )


def run_validation_eval():
    ks_val = KS(L=L, N=x.size, a_dim=A_DIM)
    dwell_steps = int(np.ceil(VAL_DWELL_TIME / ks_val.dt))
    if dwell_steps <= 0:
        raise ValueError(f"Invalid dwell steps computed from VAL_DWELL_TIME={VAL_DWELL_TIME} and dt={ks_val.dt}")
    if not (0.0 < VAL_FINAL_WINDOW_FRAC <= 1.0):
        raise ValueError(f"VAL_FINAL_WINDOW_FRAC must be in (0,1], got {VAL_FINAL_WINDOW_FRAC}")
    final_window = max(1, int(np.ceil(VAL_FINAL_WINDOW_FRAC * VAL_MAX_STEPS)))
    final_start = VAL_MAX_STEPS - final_window
    epsilon = float(VAL_EPSILON_BETA * np.linalg.norm(U_bf))

    successes = 0
    final_errors = np.zeros(validation_rows.size, dtype=np.float32)

    with torch.no_grad():
        for k, row_idx in enumerate(validation_rows):
            obs = np.float32(init_states[int(row_idx)].copy())
            error_curve = np.zeros(VAL_MAX_STEPS, dtype=np.float32)
            for t in range(VAL_MAX_STEPS):
                state = np.float32(obs[sensor_indices])
                action = trainer.get_action(state, Test=True)
                obs = ks_val.advance(obs, action)
                error_curve[t] = np.float32(np.linalg.norm(obs - U_bf))
            final_errors[k] = np.float32(np.mean(error_curve[final_start:]))
            if first_stable_step(error_curve, epsilon=epsilon, dwell_steps=dwell_steps) is not None:
                successes += 1

    return float(successes / validation_rows.size), float(np.mean(final_errors))


validation_rows = np.array([], dtype=np.int64)
validation_rows_path = os.path.join(MODEL_DIR, "validation_rows.json")
validation_history_path = os.path.join(MODEL_DIR, "validation_history.csv")
best_meta_path = os.path.join(MODEL_DIR, "best_checkpoint_meta.json")

baseline_final_error = None
best_score = -np.inf
best_score_episode = -1
best_success_rate = None
best_mean_final_error = None
no_improve_eval_count = 0
early_stop_triggered = False
early_stop_reason = ""

if EARLY_STOP_ENABLED and not Test:
    if VAL_SIZE <= 0:
        raise ValueError(f"VAL_SIZE must be positive, got {VAL_SIZE}")
    if VAL_SIZE > init_states.shape[0]:
        raise ValueError(
            f"VAL_SIZE ({VAL_SIZE}) exceeds rows in {VAL_INIT_FILE} ({init_states.shape[0]})."
        )
    if VAL_MAX_STEPS <= 0:
        raise ValueError(f"VAL_MAX_STEPS must be positive, got {VAL_MAX_STEPS}")
    if EVAL_INTERVAL <= 0:
        raise ValueError(f"EVAL_INTERVAL must be positive, got {EVAL_INTERVAL}")
    if PATIENCE_EVALS <= 0:
        raise ValueError(f"PATIENCE_EVALS must be positive, got {PATIENCE_EVALS}")

    rng = np.random.default_rng(VAL_SEED)
    validation_rows = np.sort(rng.choice(init_states.shape[0], size=VAL_SIZE, replace=False).astype(np.int64))

    validation_rows_payload = {
        "seed": int(VAL_SEED),
        "init_file": VAL_INIT_FILE,
        "val_size": int(VAL_SIZE),
        "rows": validation_rows.tolist(),
        "val_max_steps": int(VAL_MAX_STEPS),
        "val_final_window_frac": float(VAL_FINAL_WINDOW_FRAC),
        "val_epsilon_beta": float(VAL_EPSILON_BETA),
        "val_dwell_time": float(VAL_DWELL_TIME),
        "eval_interval": int(EVAL_INTERVAL),
        "min_episodes": int(MIN_EPISODES),
        "patience_evals": int(PATIENCE_EVALS),
        "delta_sr": float(DELTA_SR),
        "delta_err": float(DELTA_ERR),
        "best_score_alpha": float(BEST_SCORE_ALPHA),
    }
    with open(validation_rows_path, 'w', encoding='utf-8') as f:
        json.dump(validation_rows_payload, f, indent=2)

    if not os.path.exists(validation_history_path):
        with open(validation_history_path, 'w', encoding='utf-8') as f:
            f.write(
                "episode,success_rate,mean_final_error,score,sr_gain,err_rel_gain,"
                "no_improve_eval_count,meaningful_improvement,best_score,best_score_episode\n"
            )

    print('Validation setup complete: size=', VAL_SIZE, 'seed=', VAL_SEED,
          'rows_file=', validation_rows_path)


try:
    for _ep in range(ini,MAX_EPISODES):
        current_ep = _ep
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

            last_observation = observation
            last_new_observation = new_observation
            
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
            _save_training_snapshot(_ep)

        if EARLY_STOP_ENABLED and not Test and ((_ep - ini + 1) % EVAL_INTERVAL == 0):
            success_rate, mean_final_error = run_validation_eval()
            if baseline_final_error is None:
                baseline_final_error = max(mean_final_error, 1e-12)

            score = success_rate - BEST_SCORE_ALPHA * (mean_final_error / max(baseline_final_error, 1e-12))

            if best_success_rate is None or best_mean_final_error is None:
                sr_gain = np.nan
                err_rel_gain = np.nan
                meaningful_improvement = True
                no_improve_eval_count = 0
                best_success_rate = success_rate
                best_mean_final_error = mean_final_error
            else:
                sr_gain = success_rate - best_success_rate
                err_rel_gain = (best_mean_final_error - mean_final_error) / max(best_mean_final_error, 1e-12)
                meaningful_improvement = (sr_gain >= DELTA_SR) or (err_rel_gain >= DELTA_ERR)
                if meaningful_improvement:
                    no_improve_eval_count = 0
                else:
                    no_improve_eval_count += 1
                best_success_rate = max(best_success_rate, success_rate)
                best_mean_final_error = min(best_mean_final_error, mean_final_error)

            if score > best_score:
                _save_training_snapshot(_ep)
                _copy_best_checkpoint_aliases(_ep)
                best_score = score
                best_score_episode = _ep
                best_meta = {
                    "episode": int(_ep),
                    "score": float(score),
                    "success_rate": float(success_rate),
                    "mean_final_error": float(mean_final_error),
                    "baseline_final_error": float(baseline_final_error),
                    "best_score_alpha": float(BEST_SCORE_ALPHA),
                    "eval_interval": int(EVAL_INTERVAL),
                    "min_episodes": int(MIN_EPISODES),
                    "patience_evals": int(PATIENCE_EVALS),
                    "delta_sr": float(DELTA_SR),
                    "delta_err": float(DELTA_ERR),
                    "val_size": int(VAL_SIZE),
                    "val_max_steps": int(VAL_MAX_STEPS),
                    "val_final_window_frac": float(VAL_FINAL_WINDOW_FRAC),
                    "val_epsilon_beta": float(VAL_EPSILON_BETA),
                    "val_dwell_time": float(VAL_DWELL_TIME),
                    "val_seed": int(VAL_SEED),
                    "val_rows_file": validation_rows_path,
                }
                with open(best_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(best_meta, f, indent=2)

            row = {
                "episode": _ep,
                "success_rate": success_rate,
                "mean_final_error": mean_final_error,
                "score": score,
                "sr_gain": 0.0 if np.isnan(sr_gain) else float(sr_gain),
                "err_rel_gain": 0.0 if np.isnan(err_rel_gain) else float(err_rel_gain),
                "no_improve_eval_count": no_improve_eval_count,
                "meaningful_improvement": meaningful_improvement,
                "best_score": float(best_score),
                "best_score_episode": int(best_score_episode),
            }
            _append_validation_history_row(validation_history_path, row)

            sr_gain_text = "n/a" if np.isnan(sr_gain) else f"{sr_gain:.4f}"
            err_gain_text = "n/a" if np.isnan(err_rel_gain) else f"{err_rel_gain:.4f}"
            print(
                "VALIDATION :- episode:", _ep,
                " success_rate:", f"{success_rate:.3f}",
                " mean_final_error:", f"{mean_final_error:.6f}",
                " score:", f"{score:.6f}",
                " sr_gain:", sr_gain_text,
                " err_rel_gain:", err_gain_text,
                " no_improve_eval_count:", no_improve_eval_count
            )

            if (_ep + 1) >= MIN_EPISODES and no_improve_eval_count >= PATIENCE_EVALS:
                early_stop_triggered = True
                early_stop_reason = (
                    f"Early stop at episode {_ep}: no meaningful improvement for "
                    f"{PATIENCE_EVALS} evaluations (interval={EVAL_INTERVAL}, "
                    f"min_episodes={MIN_EPISODES})."
                )
                print(early_stop_reason)
                break

except KeyboardInterrupt:
    print('\nKeyboardInterrupt received. Stopping training early.')
    if not Test and current_ep is not None:
        print(f'Saving interruption checkpoint at episode {current_ep}...')
        _save_training_snapshot(current_ep)
    raise SystemExit(130)


if early_stop_triggered:
    print('Training ended by early stopping.')
    if best_score_episode >= 0:
        print('Best checkpoint episode:', best_score_episode, 'score:', np.float32(best_score))
        print('Best checkpoint metadata:', best_meta_path)
else:
    print('Completed episodes')
