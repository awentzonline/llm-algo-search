import ale_py
import gymnasium as gym
import numpy as np
import torch


class RLPixelsEvaluator:
    def evaluate(self, cfg, agent_cls):
        gym.register_envs(ale_py)

        env = gym.make(cfg.env_name)
        obs_shape = env.observation_space.shape
        action_dims = env.action_space.n
        agent = agent_cls(cfg.model_dims, obs_shape, action_dims)
        agent = agent.to(cfg.device)
        mean_episode_rewards = []
        max_episode_rewards = []
        max_steps = 0
        episode_steps = []
        for episode_i in range(cfg.num_episodes):
            render_mode = 'human' if episode_i > 99 and episode_i % 100 == 0 else 'rgb_array'
            env = gym.make(cfg.env_name, render_mode=render_mode)
            obs, _ = env.reset()
            reward = 0.
            agent.start_episode()
            obs = agent.preprocess_obs(obs)
            all_obs, all_actions, all_rewards = [obs.detach().cpu()], [0], [0]
            for step_i in range(cfg.max_steps):
                action = agent.policy(
                    obs.to(cfg.device),
                    torch.FloatTensor([reward]).to(cfg.device),
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                obs = agent.preprocess_obs(obs)
                all_obs.append(obs.detach().cpu())
                all_actions.append(action)
                all_rewards.append(reward)
                if terminated or truncated:
                    break

            print((np.mean(all_rewards)), len(all_rewards))
            max_steps = max(max_steps, step_i)
            episode_steps.append(step_i)
            mean_episode_rewards.append(np.mean(all_rewards))
            max_episode_rewards.append(max(all_rewards))
            all_obs = torch.stack(all_obs).to(cfg.device)
            all_actions = torch.from_numpy(np.stack(all_actions)).long().to(cfg.device)
            all_rewards = torch.from_numpy(np.stack(all_rewards, dtype=np.float32)).to(cfg.device)
            agent.train_on_episode(all_obs, all_actions, all_rewards)

        return dict(
            mean_max_reward=np.mean(max_episode_rewards),
            mean_reward=np.mean(mean_episode_rewards[-3:]),
            max_steps=max_steps,
            mean_steps=np.mean(episode_steps),
        )
