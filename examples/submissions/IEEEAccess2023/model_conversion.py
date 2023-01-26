import gym
import torch

from stable_baselines3 import SAC

import glob
import pynever.pytorch_layers as pyt_layers
import pynever.strategies.conversion as conversion
from datetime import datetime
import copy
import onnx

from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from stable_baselines3.common.evaluation import evaluate_policy

import shared_constants

model_dir = "rl_models/"
model_dir_paths = glob.glob(f"{model_dir}*")

id_idx = 0
for path in model_dir_paths:

    id_arch_dict = shared_constants.id_arch_dict
    for key, value in id_arch_dict.items():
        print(key, value)
        if value in path:
            network_id = key

    print(path, network_id)

    best_model_path = path + "/best_model.zip"

    env_name = "hover-aviary-v0"
    obs = ObservationType.KIN
    act = ActionType.ONE_D_RPM
    model = SAC.load(best_model_path)

    eval_env = gym.make(env_name,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=obs,
                        act=act
                        )
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print(mean_reward, std_reward)

    with open("rl_results.txt", "a+") as res_f:
        res_f.write(f"{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')},{network_id},{mean_reward},{std_reward}\n")
    id_idx += 1

    layers = []
    # First part of actor network
    for m in model.actor.latent_pi.modules():
        if not isinstance(m, torch.nn.Sequential):
            layers.append(m)

    # Second part of actor network
    layers.append(model.actor.mu)

    in_dim = (12,)
    wrap_layers = []
    layer_number = 1
    for m in layers:

        if isinstance(m, torch.nn.Linear):
            if m.bias is None:
                has_bias = False
            else:
                has_bias = True
            new_layer = pyt_layers.Linear(f"Linear_{layer_number}", in_dim, out_dim=(m.out_features,),
                                          in_features=m.in_features, out_features=m.out_features, bias=has_bias)

            new_layer.weight = copy.deepcopy(m.weight)
            if has_bias:
                new_layer.bias = copy.deepcopy(m.bias)

            wrap_layers.append(new_layer)
            in_dim = (m.out_features,)

        elif isinstance(m, torch.nn.ReLU):
            new_layer = pyt_layers.ReLU(f"ReLU_{layer_number}", in_dim, in_dim)
            layer_number += 1
            wrap_layers.append(new_layer)

    #network_id = path.replace(model_dir, '')
    model.actor.latent_pi = pyt_layers.Sequential(network_id, "X", wrap_layers[0:-1])
    model.actor.mu = wrap_layers[-1]

    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )

    print(mean_reward, std_reward)

    pyt_net = pyt_layers.Sequential(network_id, "X", wrap_layers)
    pynever_net = conversion.PyTorchConverter().to_neural_network(conversion.PyTorchNetwork(network_id, pyt_net))
    onnx_net = conversion.ONNXConverter().from_neural_network(pynever_net)
    onnx.save(onnx_net.onnx_network, f"onnx_nets/{network_id}.onnx")