import gym

env = gym.make("CartPole-v0")
total_reward = 0.0
steps = 0
env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1
    if done:
        break

print(f"Done game in {steps} steps with reward {total_reward}")
