import gymnasium as gym
import numpy as np
import torch


class RLEvaluator:
    def evaluate(self, cfg, agent_cls):
        env = gym.make(cfg.env_name)
        obs_dims = np.prod(env.observation_space.shape)
        action_dims = env.action_space.n
        agent = agent_cls(cfg.model_dims, obs_dims, action_dims)
        agent = agent.to(cfg.device)
        mean_episode_rewards = []
        max_episode_rewards = []
        max_steps = 0
        episode_steps = []
        for episode_i in range(cfg.num_episodes):
            obs, _ = env.reset()
            reward = 0.
            agent.start_episode()
            all_obs, all_actions, all_rewards = [obs], [0], [0]
            for step_i in range(cfg.max_steps):
                action = agent.policy(
                    torch.from_numpy(obs).float().to(cfg.device),
                    torch.FloatTensor([reward]).to(cfg.device),
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                all_obs.append(obs)
                all_actions.append(action)
                all_rewards.append(reward)
                if terminated or truncated:
                    break

            print((np.mean(all_rewards)), len(all_rewards))
            max_steps = max(max_steps, step_i)
            episode_steps.append(step_i)
            mean_episode_rewards.append(np.mean(all_rewards))
            max_episode_rewards.append(max(all_rewards))
            all_obs = torch.from_numpy(np.stack(all_obs, dtype=np.float32)).to(cfg.device)
            all_actions = torch.from_numpy(np.stack(all_actions)).long().to(cfg.device)
            all_rewards = torch.from_numpy(np.stack(all_rewards, dtype=np.float32)).to(cfg.device)
            agent.train_on_episode(all_obs, all_actions, all_rewards)

        return dict(
            mean_max_reward=np.mean(max_episode_rewards),
            mean_reward=np.mean(mean_episode_rewards[-3:]),
            max_steps=max_steps,
            mean_steps=np.mean(episode_steps),
        )
