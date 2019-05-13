import numpy as np
import gym

# Функция уменьшения эпсилон
def decay(eps, Q, state, episode):
    if np.random.rand() > eps:
        action = np.argmax(Q[state] + np.random.randn(1, n_actions) / (episode / 4))
    else:
        action = env.action_space.sample()
        eps -= 10 ** -5
    return action, eps

# Функция для запуска игры на основе обученных данных
def start(Q, Episodes, Steps):
    for episode in range(1, Episodes + 1):
        state = env.reset()

        for step in range(1, Steps):

            action = np.argmax(Q[state])
            state2, reward, done, info = env.step(action)
            state = state2
            env.render()
            if done == True:
                break

# Инициализируем среду
env = gym.make("FrozenLake-v0")
n_states = env.observation_space.n # 16
n_actions = env.action_space.n # 4
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Инициализируем переменные, необходимые для обучения
alpha, gamma, eps, Episodes, Steps, reward, rewardTracker = 0.8, 0.95, 1, 5000, 300, None, []

# Обучаем
for episode in range(1, Episodes + 1):
    G = 0
    state = env.reset()

    for step in range(1, Steps):
         action, eps = decay(eps, Q, state, episode)
         state2, reward, done, info = env.step(action)
         Q[state, action] += alpha * (reward + gamma * np.max(Q[state2]) - Q[state, action])
         state = state2
         G += reward

    # Добавляем награду после каждого эпизода для отслеживания успехов
    rewardTracker.append(G)
    if episode % 500 == 0 or episode == 1:
        print('Alpha {}  Gamma {}  Epsilon {:04.3f}  Эпизод {} из {}'.format(alpha, gamma, eps, episode, Episodes))
        print("Средняя награда: {}".format(sum(rewardTracker) / episode))

# Запускаем игру
start(Q, 1, 300)