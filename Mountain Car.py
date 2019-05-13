import numpy as np
import gym


def decay(epsilon, Q, min_eps, reduction):
    if np.random.random() < 1 - epsilon:
        action = np.argmax(Q[state_agent[0], state_agent[1]])
    else:
        action = np.random.randint(0, env.action_space.n)

    if epsilon > min_eps:
        epsilon -= reduction

    return action, epsilon


env = gym.make('MountainCar-v0')
env.reset()

# Определяем переменные для обучения
episodes, min_eps, alpha, gamma, epsilon, G = 5000, 0, 0.2, 0.9, 0.8, 0

# Определяем целочисленное пространство
num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1

# Создаем Q массив (трехмерный массив 15x19x3)
Q = np.zeros([num_states[0], num_states[1], env.action_space.n])

# Расчитаем велечеину для уменьшения эпсилон
reduction = (epsilon - min_eps) / episodes

# Обучаем
for episode in range(episodes):

    done = False
    G = 0
    state = env.reset()

    # переводим в целое значение state
    state_agent = (state - env.observation_space.low) * np.array([10, 100])
    state_agent = np.round(state_agent, 0).astype(int)

    while done != True:
        # Запускаем среду для 20 последних эпизодов
        if episode >= (episodes - 20):
            env.render()

        # Определяем след.действия либо случайно либо на основе Q
        action, epsilon = decay(epsilon, Q, min_eps, reduction)

        # Получаем след,шаг и информацию
        state2, reward, done, info = env.step(action)

        # переводим в целое значение state2
        state2_agent = (state2 - env.observation_space.low) * np.array([10, 100])
        state2_agent = np.round(state2_agent, 0).astype(int)

        # Если агент добрался
        if done == True and state2[0] >= 0.5:

            Q[state_agent[0], state_agent[1], action] = reward

        # Если не добрался, меняем значение
        else:
            delta = alpha * (reward +
                             gamma * np.max(Q[state2_agent[0],
                                              state2_agent[1]]) -
                             Q[state_agent[0], state_agent[1], action])
            Q[state_agent[0], state_agent[1], action] += delta

        # считаем награду
        G += reward
        state_agent = state2_agent

    if episode % 500 == 0:
        print('Эпизод {} награда: {}'.format(episode, G))
        if episode > 4500:
            break
    if episode % 100 == 0 and episode > 4500:
        print('Эпизод {} награда: {}'.format(episode, G))
env.close()
