import os
from datetime import datetime
import numpy as np
import gym
import torch
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

import shared_constants

EPISODE_REWARD_THRESHOLD = -0 # Upperbound: rewards are always negative, but non-zero
"""float: Reward threshold to halt the script."""

env = 'hover'
algo = 'sac'
obs = ObservationType('kin')
act = ActionType('one_d_rpm')
cpu = 1
steps = 50000
output_folder = 'rl_models'
actor_archs = [[32, 16, 8], [64, 32, 16], [128, 64, 32], [256, 128, 64], [32, 16], [64, 32], [128, 64], [256, 128]]
critic_arch = [512, 256, 128, 64]

learning_rate = 0.001

for actor_arch in actor_archs:
    #### Save directory ########################################
    filename = os.path.join(output_folder, 'save-'+env+'-'+algo+'-'+obs.value+'-'+act.value+'-'+actor_arch.__str__()+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    env_name = env+"-aviary-v0"
    sa_env_kwargs = dict(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=obs, act=act)

    train_env = make_vec_env(HoverAviary, env_kwargs=sa_env_kwargs, n_envs=cpu, seed=0)

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(qf=critic_arch, pi=actor_arch))

    model = SAC(sacMlpPolicy, train_env, policy_kwargs=offpolicy_kwargs,
                tensorboard_log=filename+'/tb/', verbose=1, learning_rate=learning_rate)

    eval_env = gym.make(env_name, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=obs, act=act)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD, verbose=1)

    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(2000/cpu),
                                 deterministic=True,
                                 render=False
                                 )

    model.learn(total_timesteps=steps,
                callback=eval_callback,
                log_interval=100,
                )

    #### Save the model ########################################
    model.save(filename+'/success_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

