import gym
import numpy as np

env = gym.make("Taxi-v2")
state = env.reset()

n_states = env.observation_space.n # 500
n_actions = env.action_space.n # 6
Q = np.zeros([n_states, n_actions])

G, episodes, alpha, gamma = 0, 2000, 0.6, 0.4

# Обучаем
for episode in range(1, episodes + 1):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done == False:
        action = np.argmax(Q[state]) # Выбираю действие с наибольшей наградой
        state2, reward, done, info = env.step(action) # Обновляюи информацию о положении и прочей ерунде
        Q[state, action] += alpha * (( reward + gamma * ( np.max(Q[state2])) - Q[state, action]))
        # Делаем шаг
        state = state2
        G += reward

    if episode % 100 == 0:
        print( "Эпизод {}, Итоговая награда: {}".format(episode, G))

# np.savetxt("C:\\Users\Юрий\PycharmProjects\OpenAI\Q table for taxi.txt",Q)
# Q = np.loadtxt("C:\\Users\Юрий\PycharmProjects\OpenAI\Q table for taxi.txt")

state = env.reset()
done = False

while done == False:
    action = np.argmax(Q[state])
    state2, reward, done, info = env.step(action)
    state = state2
    env.render()