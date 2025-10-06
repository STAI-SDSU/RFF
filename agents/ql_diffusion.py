# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

    def actor(self, state):
        return self.actor.sample(state)

    def critic(self, state, action):
        return self.critic(state, action)

    def critic_q_min(self, state, action, target=True):
        if target:
            return self.critic_target.q_min(state, action).flatten()
        else:
            return self.critic.q_min(state, action).flatten()

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None, multi_task=False):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for iter in range(iterations):
            if (iter % 100 == 0):
                print(f"Iteration {iter} for current epoch.")
            # Sample replay buffer / batch
            if multi_task:
                state, action, next_state, reward, not_done, task_id = replay_buffer.sample(batch_size)
                state = torch.cat([state, task_id], dim=1)
                next_state = torch.cat([next_state, task_id], dim=1)
            else:
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            #print(f"State shape: {state.shape}")
            #print(f"Action shape: {action.shape}")
            #print(f"First state: {state[0]}")
            #exit()

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            # Q target loss explodes over time. Testing clipping
            # to see if this can prevent that issue
            target_q = target_q.clamp(min=-10, max=10)

            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC L/ss', bc_loss.item(), self.step)
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    # Untrain loop to untrain 1 task from a dataset
    def untrain(self, replay_buffer, iterations, batch_size=100, log_writer=None, multi_task=False, exp_samples=100, lam_forget=5, beta_fisher=0.95, eta_forget=1, eps_fisher=3e-3, lr_forget=4e-7, lr_retain=3e-4, lam_diffusion=3e-2, use_g_retain=True):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        # Initialize the Fisher Information Matrix (FIM)
        #fisher = {name: torch.zeros_like(p) for name, p in self.actor.named_parameters() if p.requires_grad}
        fisher = {name: torch.ones_like(p) for name, p in self.actor.named_parameters() if p.requires_grad}
        #total_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        #fisher = {name: torch.full_like(p, 1.0 / total_params) for name, p in self.actor.named_parameters() if p.requires_grad}
        for iter in range(iterations):
            if (iter % 10 == 0):
                print(f"Iteration {iter} for current epoch.")
            # Sample replay buffer / batch
            if multi_task:
                state, action, next_state, reward, not_done, task_id, indicator = replay_buffer.sample(batch_size)
                state = torch.cat([state, task_id], dim=1)
                next_state = torch.cat([next_state, task_id], dim=1)
            else:
                state, action, next_state, reward, not_done, indicator = replay_buffer.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            # Q target loss explodes over time. Testing clipping
            # to see if this can prevent that issue
            # 10 is just an arbitrary value that worked for me
            #target_q = target_q.clamp(min=-10, max=10)

            # Calculate the expected Q_targets
            pred_q1s = []
            pred_q2s = []
            for sample in range(exp_samples):
                pred_actions = self.actor(state)
                q_pi_1, q_pi_2 = self.critic_target(state, pred_actions)
                pred_q1s.append(q_pi_1)
                pred_q2s.append(q_pi_2)
            # Stack the tensors
            pred_q1s      = torch.stack(pred_q1s, dim=0)
            pred_q2s      = torch.stack(pred_q2s, dim=0)
            expected_q1s  = pred_q1s.mean(dim=0)
            expected_q2s  = pred_q2s.mean(dim=0)
            # Create the expected Q target
            exp_q_target  = torch.min(expected_q1s, expected_q2s)
            # Calculate target q value
            y_forget = (reward + (not_done * self.discount * target_q) - (lam_forget * exp_q_target * indicator)).detach()

            critic_loss = F.mse_loss(current_q1, y_forget) + F.mse_loss(current_q2, y_forget)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=1)
            self.critic_optimizer.step()

            """ Policy Training """
            forget_mask = (indicator == 1).squeeze()
            retain_mask = (indicator == 0).squeeze()
            forget_state = state[forget_mask]
            forget_action = action[forget_mask]
            new_action = self.actor(forget_state)
            #q1_new_action, q2_new_action = self.critic(state, new_action)

            #q_val = torch.min(q1_new_action, q2_new_action)
            # Output of losses main shape is the number of timesteps by batch size
            actor_losses = self.actor.t_loss(forget_action, forget_state, q_func=self.critic, lam=lam_diffusion)

            # First mean is the expected loss grad per sample. Second is across batch
            per_sample = torch.stack(actor_losses, dim=0).mean(0).mean()

            self.actor.zero_grad(set_to_none=True)
            per_sample.backward()           # grads now equal Σ_i ∇θ E_{t,ε}[ℓ_i]

            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)

            neb_naff = {name: torch.zeros_like(p) for name, p in self.actor.named_parameters() if p.requires_grad}
            forget_len = len(forget_state)
            # Accumulate into g_forget across batches
            for name, p in self.actor.named_parameters():
                if p.grad is not None:
                    # Sum is just expected value (mean) * num samples (forget batch size)
                    neb_naff[name] += p.grad.detach() * forget_len

            # Update the FIM
            for name, p in self.actor.named_parameters():
                if p.grad is not None:
                    fisher[name] = beta_fisher * fisher[name] + (1 - beta_fisher) * (p.grad.detach() ** 2)
                    fisher[name].clamp_(min=1e-6, max=0.25e3) # clamping to make sure no param is over-weighted for FIM

            # Update the actors parameters
            #temp=0
            with torch.no_grad():
                for name, param in self.actor.named_parameters():
                    if name in neb_naff and name in fisher:
                        step = neb_naff[name] / (fisher[name] + eps_fisher)
                        # Need to scale the updates since the loss is quite aggressive
                        param += eta_forget * step * lr_forget

            if use_g_retain:
                # ---- Retain repair step (standard optimizer) ----
                retain_state = state[retain_mask]
                retain_action = action[retain_mask]
                bc_loss = self.actor.loss(retain_action, retain_state)
                new_action = self.actor(retain_state)
                q1, q2 = self.critic(retain_state, new_action)
                if np.random.uniform() > 0.5:
                    q_loss = - q1.mean() / (q2.abs().mean().detach() + 1e-8)
                else:
                    q_loss = - q2.mean() / (q1.abs().mean().detach() + 1e-8)

                actor_loss = bc_loss + self.eta * q_loss
                self.actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.grad_norm > 0: 
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)

                # Store the losses
                g_retain = {name: torch.zeros_like(p) for name, p in self.actor.named_parameters() if p.requires_grad}
                for name, p in self.actor.named_parameters():
                    if p.grad is not None:
                        # Sum is just expected value (mean) * num samples (forget batch size)
                        g_retain[name] += p.grad.detach()
                # Update the critic parameters
                with torch.no_grad():
                    for name, param in self.actor.named_parameters():
                            step = g_retain[name]
                            # Also need to scale the retain updates to be more subtle
                            param -= eta_forget * step * lr_retain

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                log_writer.add_scalar('Actor Loss', actor_losses[-1].mean().item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_losses[-1].mean().item())
            metric['bc_loss'].append(0.)
            metric['ql_loss'].append(0.)
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.critic_lr_scheduler.step()

        return metric

    def reward_tune(self, replay_buffer, iterations, batch_size=100, log_writer=None, multi_task=False, rand_reward=False):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for iter in range(iterations):
            if (iter % 100 == 0):
                print(f"Iteration {iter} for current epoch.")
            # Sample replay buffer / batch
            if multi_task:
                state, action, next_state, reward, not_done, task_id, indicator = replay_buffer.sample(batch_size)
                state = torch.cat([state, task_id], dim=1)
                next_state = torch.cat([next_state, task_id], dim=1)
            else:
                state, action, next_state, reward, not_done, indicator = replay_buffer.sample(batch_size)

            forget_mask = (indicator == 1).squeeze()
            # Assign a random reward for the forget tasks
            if rand_reward:
                for i in range(len(reward)):
                    if forget_mask[i]:
                        rand_sample = reward[i] * np.random.uniform(0, 1)
                        reward[i] -= rand_sample # guarantee not too large of an update
            else:
                # Can change this to finetune while removing the original forget samples.
                # My take is to do said finetuning, just make the forget set a negative 
                # reward instead of just removing it completely.
                reward[forget_mask] = -1 * reward[forget_mask]

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            # Q target loss explodes over time. Testing clipping
            # to see if this can prevent that issue
            target_q = target_q.clamp(min=-10, max=10)

            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    # Untrain loop for EraseDiff algorithm
    def erase_diff(self, replay_buffer, iterations, batch_size=100, log_writer=None, multi_task=False, lr=3e-5, k=10, eta=1e-4):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for iter in range(iterations):
            if (iter % 10 == 0):
                print(f"Iteration {iter} for current epoch.")
            # Sample replay buffer / batch
            if multi_task:
                state, action, next_state, reward, not_done, task_id, indicator = replay_buffer.sample(batch_size)
                state = torch.cat([state, task_id], dim=1)
                next_state = torch.cat([next_state, task_id], dim=1)
            else:
                state, action, next_state, reward, not_done, indicator = replay_buffer.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            # Q target loss explodes over time. Testing clipping
            # to see if this can prevent that issue
            # 10 is just an arbitrary value that worked for me
            #target_q = target_q.clamp(min=-10, max=10)

            # Calculate the expected Q_targets
            pred_actions = self.actor(state)
            q1, q2 = self.critic_target(state, pred_actions)
            # Create the expected Q target
            q_target  = torch.min(q1, q2)
            # Calculate target q value
            y_forget = (reward + (not_done * self.discount * target_q)).detach()

            critic_loss = F.mse_loss(current_q1, y_forget) + F.mse_loss(current_q2, y_forget)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=1)
            self.critic_optimizer.step()

            """ Policy Training """
            forget_mask = (indicator == 1).squeeze()
            retain_mask = (indicator == 0).squeeze()
            forget_state = state[forget_mask]
            forget_action = action[forget_mask]
            retain_state = state[retain_mask]
            retain_action = action[retain_mask]
            new_action = self.actor(forget_state)

            # Create phi copy of original params
            #phi_k = {name: p for name, p in self.actor.named_parameters() if p.requires_grad}
            phi_0 = copy.deepcopy(self.actor)
            phi_k = copy.deepcopy(self.actor)
            for _ in range(k):
                # Output of losses main shape is the number of timesteps by batch size
                phi_losses = phi_k.loss(forget_action, forget_state)
                per_sample = phi_losses.mean()

                phi_k.zero_grad(set_to_none=True)
                per_sample.backward()           # grads now equal Σ_i ∇θ E_{t,ε}[ℓ_i]

                if self.grad_norm > 0: 
                    actor_grad_norms = nn.utils.clip_grad_norm_(phi_k.parameters(), max_norm=self.grad_norm, norm_type=2)

                # K steps of Gradient Descent
                with torch.no_grad():
                    for name, p in phi_k.named_parameters():
                        if p.grad is not None:
                            p.data -= lr * p.grad.detach() # use 3e-5 as lr for now

            # Compute the forget losses for phi_k and the original actor
            phi_k_losses = phi_k.loss(forget_action, forget_state)
            phi_0_losses = phi_0.loss(forget_action, forget_state)
            phi_k_loss   = phi_k_losses.mean()
            phi_0_loss   = phi_0_losses.mean()

            # Compute the respective gradients
            phi_k.zero_grad(set_to_none=True)
            phi_0.zero_grad(set_to_none=True)
            phi_k_loss.backward()
            phi_0_loss.backward()

            # Find the retain losses for the original params
            actor_losses = self.actor.loss(retain_action, retain_state)
            actor_loss   = actor_losses.mean()
            self.actor.zero_grad(set_to_none=True)
            actor_loss.backward()

            # Pre-compute the l2 for the grad of g(theta_t)
            grads_g = []   # collect flattened grads
            for (_, p0), (_, pk) in zip(phi_0.named_parameters(), phi_k.named_parameters()):
                if (p0.grad is not None) and (pk.grad is not None):
                    diff = (p0.grad.detach() - pk.grad.detach()).reshape(-1)
                    grads_g.append(diff)
            # Concatenate into a single vector
            grad_g_vec = torch.cat(grads_g)
            # Compute squared L2 norm
            norm_sq = (grad_g_vec ** 2).sum()

            a_t = eta * norm_sq

            # Pre-compute lamba_t
            retain_grads = []
            for _, a in self.actor.named_parameters():
                if a.grad is not None:
                    retain_grads.append(a.grad.detach().reshape(-1))
            grad_r_vec = torch.cat(retain_grads)
            dot = (grad_g_vec @ grad_r_vec)
            lambda_t = torch.maximum(
                torch.tensor(0.0, device=norm_sq.device),
                (a_t - dot) / norm_sq
            )

            # Update the actor's parameters
            with torch.no_grad():
                for (name_pk, pk), (name_p0, p0), (name_a, a) in zip(phi_k.named_parameters(), phi_0.named_parameters(), self.actor.named_parameters()):
                    if (pk.grad is not None) and (p0.grad is not None) and (a.grad is not None):
                        a.data -= eta * (a.grad.detach() + lambda_t * (p0.grad.detach() - pk.grad.detach()))

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                log_writer.add_scalar('Actor Retain Loss', actor_loss.item(), self.step)
                log_writer.add_scalar('Actor Forget Loss', phi_0_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(0.)
            metric['ql_loss'].append(0.)
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.critic_lr_scheduler.step()

        return metric

    # Untrain loop to untrain 1 task from a dataset
    def traj_deleter(self, replay_buffer, iterations, batch_size=100, log_writer=None, multi_task=False, exp_samples=100, lam_forget=2):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        # Initialize the Fisher Information Matrix (FIM)
        #fisher = {name: torch.zeros_like(p) for name, p in self.actor.named_parameters() if p.requires_grad}
        fisher = {name: torch.ones_like(p) for name, p in self.actor.named_parameters() if p.requires_grad}
        #total_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        #fisher = {name: torch.full_like(p, 1.0 / total_params) for name, p in self.actor.named_parameters() if p.requires_grad}
        for iter in range(iterations):
            if (iter % 10 == 0):
                print(f"Iteration {iter} for current epoch.")
            # Sample replay buffer / batch
            if multi_task:
                state, action, next_state, reward, not_done, task_id, indicator = replay_buffer.sample(batch_size)
                state = torch.cat([state, task_id], dim=1)
                next_state = torch.cat([next_state, task_id], dim=1)
            else:
                state, action, next_state, reward, not_done, indicator = replay_buffer.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            # Q target loss explodes over time. Testing clipping
            # to see if this can prevent that issue
            # 10 is just an arbitrary value that worked for me
            #target_q = target_q.clamp(min=-10, max=10)

            # Calculate the expected Q_targets
            pred_q1s = []
            pred_q2s = []
            for sample in range(exp_samples):
                pred_actions = self.actor(state)
                q_pi_1, q_pi_2 = self.critic_target(state, pred_actions)
                pred_q1s.append(q_pi_1)
                pred_q2s.append(q_pi_2)
            # Stack the tensors
            pred_q1s      = torch.stack(pred_q1s, dim=0)
            pred_q2s      = torch.stack(pred_q2s, dim=0)
            expected_q1s  = pred_q1s.mean(dim=0)
            expected_q2s  = pred_q2s.mean(dim=0)
            # Create the expected Q target
            exp_q_target  = torch.min(expected_q1s, expected_q2s)
            # Calculate target q value
            y_forget = (reward + (not_done * self.discount * target_q) - (lam_forget * exp_q_target * indicator)).detach()

            critic_loss = F.mse_loss(current_q1, y_forget) + F.mse_loss(current_q2, y_forget)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=1)
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                log_writer.add_scalar('Actor Loss', actor_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(0.)
            metric['ql_loss'].append(0.)
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
            #self.actor = torch.load(f'{dir}/actor_{id}.pth')
            #self.critic = torch.load(f'{dir}/critic_{id}.pth')
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


