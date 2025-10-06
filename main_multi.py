# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gym
import numpy as np
import os
import torch
import json
import pickle

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler, Unlearn_Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter
from cheetah_velocities import cheetah_velocities
from envs.mujoco_control_envs.mujoco_control_envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv

hyperparameters = {
    'halfcheetah-medium-replay-v2':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 5, 'num_epochs': 500, 'gn': 9.0,  'top_k': 1},
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 5, 'num_epochs': 500, 'gn': 9.0,  'top_k': 1},
    #'halfcheetah-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 5, 'num_epochs': 250, 'gn': 5.0,  'top_k': 25, 'multi_task': True, 'trainset_folder': "cheetah_vel/", 'trainset_name': "cheetah_vel-*-expert", 'unlearn_task': 0, 'cheetah_type': 'vel'},
    'halfcheetah-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 5, 'num_epochs': 75, 'gn': 5.0,  'top_k': 2, 'multi_task': True, 'trainset_folder': "cheetah_dir/", 'trainset_name': "cheetah_dir-*-expert", 'unlearn_task': 0, 'cheetah_type': 'dir'}
}

def train_agent(env, env_id, state_dim, action_dim, max_action, device, output_dir, args, skip_eval=False, retrain=False, finetune=False, rand_reward=False, traj_deleter=False, erase_diff=False, per_data=1):
    ul_bool = args.unlearn or finetune or rand_reward or erase_diff or traj_deleter

    # Load dataset from pkl
    data_path = '/data/PromptDT_data/data/'
    # Create the dataset dictionary
    dataset = {}
    dataset['observations'] = []
    dataset['actions'] = []
    dataset['next_observations'] = []
    dataset['rewards'] = []
    dataset['terminals'] = []
    dataset['task_id'] = []

    if args.cheetah_type == 'dir':
        mod = 1 # used to indicate which index/data to load
        num_tasks = 2
        print("Training for cheetah-dir!")
    else:
        mod = 4 # used to indicate which index/data to load
        num_tasks = 40
        print("Training for cheetah-vel!")

    if (args.multi_task):
        print("Training on a multi-task dataset: " + args.trainset_folder)
        if (ul_bool):
            print(f"Unlearning task {args.unlearn_task}'s data!")
            dataset['indicator'] = []
        data_path += args.trainset_folder
        dataset_path = args.trainset_name+'.pkl'
        for i in range(num_tasks):
            if i%mod == 0:
                dataset_path = dataset_path.replace('*', f'{i}')
                with open(data_path+dataset_path, 'rb') as file:
                    datalist = pickle.load(file)
                # Since data is segmented into a list of dictionaries,
                # append each dictionary by key.
                for entry in datalist:
                    for key in dataset:
                        if (ul_bool):
                            if key == 'indicator':
                                # Have to create 200x1 chunks since the PromptDT
                                # data set is fragmented as such
                                # Forget first task
                                if (i==args.unlearn_task): dataset[key].append(np.ones((200, 1)))
                                else:                      dataset[key].append(np.zeros((200, 1)))
                            elif key == 'task_id':
                                if args.cheetah_type == 'vel':
                                    hot_list = [0] * 10
                                    hot_list[int(i/mod)] = 100
                                    dataset[key].append(np.tile(np.array(hot_list), (200,1)))
                                else:
                                    # I don't know why i did -100 and 1 but it was more than likely
                                    # a relational problem (i believe the point of doing so was to 
                                    # create some distance between points rather than use 1/0 which
                                    # lie closer to one another). From what i remember, using -100 and
                                    # 1 led to better overall performance.
                                    if (i==1): dataset[key].append(np.tile(np.array([0, -100]), (200, 1)))
                                    else:      dataset[key].append(np.tile(np.array([1, 0]), (200, 1)))
                            else:
                                dataset[key].append(entry[key])
                        else:
                            if not (retrain and (i==args.unlearn_task)):
                                if key == 'task_id':
                                    if args.cheetah_type == 'vel':
                                        hot_list = [0] * 10
                                        hot_list[int(i/mod)] = 100
                                        dataset[key].append(np.tile(np.array(hot_list), (200,1)))
                                    else:
                                        if (i==1): dataset[key].append(np.tile(np.array([0, -100]), (200, 1)))
                                        else:      dataset[key].append(np.tile(np.array([1, 0]), (200, 1)))
                                else:
                                    dataset[key].append(entry[key])

    else:
        dataset_path = env_id+'.pkl'
        with open(data_path+dataset_path, 'rb') as file:
            datalist = pickle.load(file)
        # Since data is segmented into a list of dictionaries,
        # append each dictionary by key.
        for entry in datalist:
            for key in dataset:
                dataset[key].append(entry[key])

    # Concat the contiguous dict's by key
    for key in dataset:
        dataset[key] = np.concatenate(dataset[key], axis=0)

    print(f"Available keys in the dataset: {dataset.keys()}")
    print(f"Length of the dataset: {len(dataset['observations'])}")
    #exit()
    if ul_bool:
        data_sampler = Unlearn_Data_Sampler(dataset, device, args.reward_tune, args.multi_task)
    else:
        data_sampler = Data_Sampler(dataset, device, args.reward_tune, args.multi_task)
    utils.print_banner('Loaded buffer')

    #if (args.unlearn):
    #    step_start_ema   = 250
    #    update_ema_every = 1
    #else:
    step_start_ema   = 1000
    update_ema_every = 5

    if args.algo == 'ql':
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      step_start_ema=step_start_ema,
                      update_ema_every=update_ema_every,
                      eta=args.eta,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn)
    elif args.algo == 'bc':
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      lr=args.lr)

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.)
    writer = None  # SummaryWriter(output_dir)

    # Test saving the model
    #agent.save_model(dir='./models/', id=env_id)
    #agent.load_model(dir='./agents_copy/cheetah-vel-pretrain/', id=env_id)
    #env_id = env_id + '_forward'

    #env_id += '_forward'

    # Pre-load the model if unlearning
    #if (args.unlearn):
    #    agent.load_model(dir='./agents_copy/cheetah-vel-finetune/', id=env_id)
    #    env_id = env_id + '_unlearn_task' + str(args.unlearn_task)
    #agent.load_model(dir='./models/', id=env_id+'_retrain_1task')
    if ul_bool:
        print("Loading a pre-trained model!")
        agent.load_model(dir='./models/multi/pretrain/'+args.cheetah_type,id=env_id)
    print("Training type:")
    print(f"Pretraining: {not ul_bool and not retrain}")
    print(f"Retraining: {retrain}")
    print(f"Our Unlearning: {args.unlearn}")
    print(f"Finetune: {finetune}")
    print(f"Random Reward: {rand_reward}")
    print(f"EraseDiff: {erase_diff}")
    print(f"Trajectory Deleter: {traj_deleter}")
    if (ul_bool):
        if erase_diff:
            if (args.multi_task):
                env_id = env_id + '_erase_diff_task' + str(args.unlearn_task)
            else:
                env_id = env_id + f'_erase_diff_{per_data}task'
        elif traj_deleter:
            if (args.multi_task):
                env_id = env_id + '_traj_deleter_task' + str(args.unlearn_task)
            else:
                env_id = env_id + f'_traj_deleter_{per_data}task'
        elif rand_reward:
            if (args.multi_task):
                env_id = env_id + '_rand_reward_task' + str(args.unlearn_task)
            else:
                env_id = env_id + f'_rand_reward_{per_data}task'
        elif finetune:
            if (args.multi_task):
                env_id = env_id + '_finetune_task' + str(args.unlearn_task)
            else:
                env_id = env_id + f'_finetune_{per_data}task'
        else:
            if (args.multi_task):
                env_id = env_id + '_unlearn_task' + str(args.unlearn_task)
            else:
                env_id = env_id + f'_unlearn_{per_data}task'
    elif (retrain):
        env_id = env_id + f'_retrain_{per_data}task'

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    print(f"Running for a maximum of {max_timesteps} timesteps.")
    metric = 100.
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)

        if ul_bool:
            if erase_diff:
                loss_metric = agent.erase_diff(data_sampler,
                                               iterations=iterations,
                                               batch_size=args.batch_size,
                                               log_writer=writer,
                                               multi_task=args.multi_task)
            elif traj_deleter:
                loss_metric = agent.traj_deleter(data_sampler,
                                                 iterations=iterations,
                                                 batch_size=args.batch_size,
                                                 log_writer=writer,
                                                 multi_task=args.multi_task)
            elif finetune or rand_reward:
                loss_metric = agent.reward_tune(data_sampler,
                                                iterations=iterations,
                                                batch_size=args.batch_size,
                                                log_writer=writer,
                                                multi_task=args.multi_task,
                                                rand_reward=rand_reward)
            else:
                loss_metric = agent.untrain(data_sampler,
                                            iterations=iterations,
                                            batch_size=args.batch_size,
                                            log_writer=writer,
                                            multi_task=args.multi_task)
        else:
            loss_metric = agent.train(data_sampler,
                                      iterations=iterations,
                                      batch_size=args.batch_size,
                                      log_writer=writer,
                                      multi_task=args.multi_task)
        training_iters += iterations
        print(f"Iteration {training_iters} out of {max_timesteps}")
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # Save the model between eval steps
        agent.save_model(dir='./models/', id=env_id)

        # Logging
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        logger.record_tabular('Trained Epochs', curr_epoch)
        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
        logger.dump_tabular()

        # Evaluation
        if args.cheetah_type == 'dir':
            eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy_dir(agent, args.env_name, args.seed,
                                                                                       eval_episodes=args.eval_episodes)
        else:
            eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy_vel(agent, args.env_name, args.seed,
                                                                                       eval_episodes=args.eval_episodes)
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                            np.mean(loss_metric['bc_loss']), np.mean(loss_metric['ql_loss']),
                            np.mean(loss_metric['actor_loss']), np.mean(loss_metric['critic_loss']),
                            curr_epoch])
        np.save(os.path.join(output_dir, "eval"), evaluations)
        logger.record_tabular('Average Episodic Reward', eval_res)
        logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
        logger.dump_tabular()

        bc_loss = np.mean(loss_metric['bc_loss'])
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)

        metric = bc_loss

        if args.save_best_model:
            agent.save_model(output_dir, env_id+curr_epoch)

    # Save the model
    agent.save_model(dir='./models/', id=env_id)

    # Model Selection: online or offline
    scores = np.array(evaluations)
    if args.ms == 'online':
        best_id = np.argmax(scores[:, 2])
        best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                    'best normalized score avg': scores[best_id, 2],
                    'best normalized score std': scores[best_id, 3],
                    'best raw score avg': scores[best_id, 0],
                    'best raw score std': scores[best_id, 1]}
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':
        bc_loss = scores[:, 4]
        top_k = min(len(bc_loss) - 1, args.top_k)
        where_k = np.argsort(bc_loss) == top_k
        best_res = {'model selection': args.ms, 'epoch': scores[where_k][0][-1],
                    'best normalized score avg': scores[where_k][0][2],
                    'best normalized score std': scores[where_k][0][3],
                    'best raw score avg': scores[where_k][0][0],
                    'best raw score std': scores[where_k][0][1]}

        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))

    # writer.close()


# Only used for cheetah direction
def eval_policy_dir(policy, env_name, seed, eval_episodes=3, max_ep_len=1000):
    directions = {
        "forward": 1.0,
        "backward": -1.0
    }
    scores = []
    for direction_name, direction_sign in directions.items():
        #eval_env = gym.make(env_name)
        eval_env = HalfCheetahDirEnv([{'direction': direction_sign}], include_goal = False)
        #eval_env = HalfCheetahDirEnv([{'direction': -1}], include_goal = False)
        eval_env.seed(seed + 100)
        if direction_sign == -1.0:
            one_hot = [0, -100]
        else:
            one_hot = [1, 0]
        for ep in range(eval_episodes):
            traj_return = 0.0
            state, done = eval_env.reset(), False
            state = np.concatenate([state, np.array(one_hot)], axis=0)
            t = 0
            while not done:
                action = policy.sample_action(np.array(state))
                xpos_before = eval_env.sim.data.qpos[0]
                state, reward, done, _ = eval_env.step(action)
                state = np.concatenate([state, np.array(one_hot)], axis=0)
                xpos_after = eval_env.sim.data.qpos[0]
                # Get torso x-velocity
                #curr_vel = eval_env.sim.data.qvel[0]
                # Reward for moving in desired direction
                curr_vel = (xpos_after - xpos_before) / eval_env.dt
                reward = direction_sign * curr_vel
                traj_return += reward
                t+=1
                if t >= max_ep_len: break
            scores.append(traj_return)
            print(f"Return for {direction_name} episode {ep}: {traj_return}")

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    #normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    normalized_scores = [np.linalg.norm(s) for s in scores]
    #avg_norm_score = eval_env.get_normalized_score(avg_reward)
    avg_norm_score = np.linalg.norm(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(
        f"Evaluation in {direction_name} direction over {eval_episodes} episodes: "
        f"avg_return={avg_reward:.2f}, norm_score={avg_norm_score:.2f}"
    )

    return avg_reward, std_reward, avg_norm_score, std_norm_score

# *** THIS IS A MODIFIED EVAL POLICY FOR ONLY CHEETAH_VEL ***
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy_vel(policy, env_name, seed, eval_episodes=1, max_ep_len=200):
    velocities = np.linspace(0.075,3,40)
    scores = []
    for i in range(40):
        if i%4==0:
            task_paths = f"../prompt-dt/config/cheetah_vel/config_cheetah_vel_task{i}.pkl"
            tasks = []
            with open(task_paths.format(i), 'rb') as f:
                task_info = pickle.load(f)
                assert len(task_info) == 1, f'Unexpected task info: {task_info}'
                tasks.append(task_info[0])
            eval_env = HalfCheetahVelEnv(tasks, include_goal = False)
            eval_env.seed(seed + 100)
            one_hot = [0] * 10
            one_hot[int(i/4)] = 100
            target_vel = velocities[i]
            for ep in range(eval_episodes):
                traj_return = 0.0
                state, done = eval_env.reset(), False
                state = np.concatenate([state, np.array(one_hot)], axis=0)
                t = 0
                while not done:
                    action = policy.sample_action(np.array(state))
                    xpos_before = eval_env.sim.data.qpos[0]
                    state, reward, done, _ = eval_env.step(action)
                    state = np.concatenate([state, np.array(one_hot)], axis=0)
                    xpos_after = eval_env.sim.data.qpos[0]
                    # Get torso x-velocity
                    #curr_vel = eval_env.sim.data.qvel[0]
                    # Reward for moving in desired direction
                    curr_vel = (xpos_after - xpos_before) / eval_env.dt
                    reward = -(curr_vel - target_vel)**2
                    traj_return += reward
                    t+=1
                    if t >= max_ep_len: break
                scores.append(traj_return)
                print(f"Return for task {i} episode {ep}: {traj_return}")

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    #normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    normalized_scores = [np.linalg.norm(s) for s in scores]
    #avg_norm_score = eval_env.get_normalized_score(avg_reward)
    avg_norm_score = np.linalg.norm(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(
        f"Evaluation average of all velocities over {eval_episodes} episodes: "
        f"avg_return={avg_reward:.2f}, norm_score={avg_norm_score:.2f}"
    )

    return avg_reward, std_reward, avg_norm_score, std_norm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="walker2d-expert-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    parser.add_argument("--seed", default=0, type=int)                         # Sets Gym, PyTorch and Numpy seeds
    #parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--num_steps_per_epoch", default=100, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql']
    parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")
    # parser.add_argument("--top_k", default=1, type=int)

    # parser.add_argument("--lr", default=3e-4, type=float)
    # parser.add_argument("--eta", default=1.0, type=float)
    # parser.add_argument("--max_q_backup", action='store_true')
    # parser.add_argument("--reward_tune", default='no', type=str)
    # parser.add_argument("--gn", default=-1.0, type=float)

    # Argument to use the multitask datasets
    parser.add_argument("--multi_task", action='store_true', help="Determine whether or not to use the Multi-Task datasets (e.g. cheetah_vel).")
    parser.add_argument("--trainset_name", default="", type=str, help="File name format of the multi-task dataset.")
    parser.add_argument("--trainset_folder", default="", type=str, help="Folder name of the multi-task dataset")
    parser.add_argument("--unlearn", action='store_true', help="Specify whether or not to unlearn the first tasks data.")
    parser.add_argument("--unlearn_task", default=0, type=int, help="Specify which task_id to unlearn.")
    parser.add_argument("--cheetah_type", default='dir', type=str, help="Specify whether you are training for cheetah dir/vel.")

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    args.num_epochs = hyperparameters[args.env_name]['num_epochs']
    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    # Since we are training on multiple different tasks, just use 1 eval episode per task
    args.eval_episodes = 1

    args.lr = hyperparameters[args.env_name]['lr']
    args.eta = hyperparameters[args.env_name]['eta']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']
    args.top_k = hyperparameters[args.env_name]['top_k']

    #args.multi_task      = hyperparameters[args.env_name]['multi_task']
    args.trainset_folder = hyperparameters[args.env_name]['trainset_folder']
    args.trainset_name   = hyperparameters[args.env_name]['trainset_name']
    #args.unlearn_task    = hyperparameters[args.env_name]['unlearn_task']
    args.cheetah_type    = hyperparameters[args.env_name]['cheetah_type']

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'

    if args.ms == 'offline': file_name += f'|k-{args.top_k}'
    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(version=f"Diffusion-Policies-RL")

    #env = gym.make(args.env_name)
    # Direction doesn't matter right now, we just need the env space
    env = HalfCheetahDirEnv([{'direction': 1}], include_goal = False)
    env_id = args.env_name

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.cheetah_type == 'dir':
        state_dim = env.observation_space.shape[0] + 2 # +2 for the direction dim
    else:
        state_dim = env.observation_space.shape[0] + 10 # +10 for the velocity dim
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env,
                env_id,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                args)
