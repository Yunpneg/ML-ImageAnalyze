import numpy as np
import time
import tkinter as tk
import matplotlib.pyplot as mlb
# import tensorflow as tf


UNIT = 100   # pixels
MAZE_H = 6  # grid height
MAZE_W = 6  # grid width
GAMMA = 0.99  # value of gamma
EPISODES = 500  # the number of episodes
EPOCHS = 10
T = 2 * MAZE_W * MAZE_H


# Create the domain
class Domain(tk.Tk, object):
    def __init__(self):
        super(Domain, self).__init__()
        self.action = ['up', 'down', 'right', 'left']
        self.a_length = len(self.action)
        self.title('My Domain')
        self._build_domain()

    # create domain
    def _build_domain(self):

        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for column in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = column, 0, column, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for row in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, row, MAZE_H * UNIT, row
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([50, 50])

        # hell_01
        hell1_center = origin + np.array([UNIT, UNIT * 4])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')

        # hell_02
        hell2_center = origin + np.array([UNIT * 2, UNIT])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # friend_01
        friend_center1 = origin + np.array([UNIT * 3, UNIT * 3])
        self.friend_01 = self.canvas.create_rectangle(
            friend_center1[0] - 15, friend_center1[1] - 15,
            friend_center1[0] + 15, friend_center1[1] + 15,
            fill='yellow')

        # create oval
        oval_center = origin + np.array([UNIT * 5, UNIT * 5])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='red')

        # create agent
        self.agent = self.canvas.create_oval(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='blue')

        # pack all
        self.canvas.pack()

    # reset agent
    def reset(self):
        self.update()
        # time.sleep(0.5)
        self.canvas.delete(self.agent)
        origin = np.array([50, 50])
        self.agent = self.canvas.create_oval(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='blue')
        # return coordination
        return self.canvas.coords(self.agent)

    @staticmethod
    def move_direction(state, action):
        '''
        Direction of agent
        :param state: current state
        :param action: current action
        :return: the direction of the agent
        '''
        base_action = np.array([0, 0])
        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if state[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        return base_action

    def agt_move_det(self, state, action):
        '''
        Deterministic version of the transition function
        :param state: current state
        :param action: current action
        :return: next state
        '''

        base_action = env.move_direction(state, action)
        # move agent
        self.canvas.move(self.agent, base_action[0], base_action[1])

        # next state
        next_s = self.canvas.coords(self.agent)

        return next_s

    def agt_move_sto(self, state, _action):
        '''
        stochastic version of the transition function
        :param state:
        :param _action:
        :return:
        '''
        if np.random.uniform() < 0.8:
            # choose best action
            action = _action
        else:
            # choose random action
            if (_action == 0) or (_action == 1): # up or down
                action = np.random.choice([2, 3])
            else:
                action = np.random.choice([0, 1])

        base_action = env.move_direction(state, action)

        # move agent
        self.canvas.move(self.agent, base_action[0], base_action[1])

        # next state
        next_s = self.canvas.coords(self.agent)

        return next_s

    # reward function
    def agt_reward(self, next_s):
        if next_s == self.canvas.coords(self.oval):
            reward = 30
            done = True
        elif next_s in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -5
            done = False
        elif next_s in [self.canvas.coords(self.friend_01)]:
            reward = 5
            done = False
        else:
            reward = 0
            done = False

        return reward, done

    # fresh the environment
    def render(self):
        # time.sleep(0.1)
        self.update()


#  Learning process
class LearningAlgo:
    # Initialize the Q_table
    def __init__(self, actions):
        self.q_table = {}
        self.actions = actions
        # self.features = features

    # Get the value of q_table
    def get_q_table(self, state, action):
        return self.q_table.get((state, action), 0.0)

    # Reset the value of q_table
    def agt_reset_value(self):
        self.q_table = {}

    def agt_choose(self, state, epsilon):
        '''
        Choose next action
        :param state: current state
        :param epsilon: current epsilon
        :return: next action
        '''
        if np.random.uniform() < epsilon:
            action = np.random.choice(self.actions)
        else:
            q = [self.get_q_table(state, a) for a in self.actions]
            max_q = max(q)
            count = q.count(max_q)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == max_q]
                i = np.random.choice(best)
            else:
                i = q.index(max_q)
            action = self.actions[i]
        return action

    def agt_learn_q(self, alpha, state, action, reward, next_s):
        '''
        Q-learning algorithm
        :param alpha: learning rate
        :param state: current state
        :param action: current action
        :param reward: reward
        :param next_s: next state
        :return:
        '''
        # next_s is not absorbing state
        max_q_new = max([self.get_q_table(next_s, a) for a in self.actions])
        value_predict = self.q_table.get((state, action))
        if value_predict is not None:
            self.q_table[(state, action)] = value_predict + alpha * ((reward + GAMMA * max_q_new) - value_predict)
        if value_predict is None:
            self.q_table[(state, action)] = reward

    def agt_learn_sarsa(self, alpha, state, action, reward, next_s, next_a):
        '''
        Sarsa algorithm
        :param alpha:
        :param state:
        :param action:
        :param reward:
        :param next_s:
        :param next_a:
        :return:
        '''
        # next_s is not absorbing state
        value_predict = self.q_table.get((state, action), None)
        value_next = self.get_q_table(next_s, next_a)

        if value_predict is not None:
            self.q_table[(state, action)] = value_predict + alpha * ((reward + GAMMA * value_next) - value_predict)
        else:
            self.q_table[(state, action)] = reward

    def agt_learn_final(self, alpha, state, action, reward):
        '''
        next state is absorbing state
        :param alpha:
        :param state:
        :param action:
        :param reward:
        :return:
        '''

        value_predict = self.q_table.get((state, action), None)
        if value_predict is not None:
            self.q_table[(state,action)] = value_predict + alpha * (reward - value_predict)
        if value_predict is None:
            self.q_table[(state, action)] = reward


# Update process
class UpDate:
    @staticmethod
    def update(_epsilon, _alpha):
        rewards = np.zeros(EPISODES)
        up_value = np.zeros(EPISODES)
        down_value = np.zeros(EPISODES)
        right_value = np.zeros(EPISODES)
        left_value = np.zeros(EPISODES)

        for epoch in range(EPOCHS):
            RL.agt_reset_value()
            for episode in range(EPISODES):
                print("###############################################")
                s = env.reset()
                learning = episode < EPISODES - 50
                if learning:
                    # eps = 1 - episode/(float(EPISODES))
                    eps = _epsilon
                else:
                    eps = 0
                cumulative_gamma = 1
                a = RL.agt_choose(str(s), eps)
                index = 0
                for timeStep in range(T):
                    # fresh env
                    env.render()
                    next_s = env.agt_move_det(s, a)
                    # next_s = env.agt_move_sto(s, a)
                    if next_s == env.canvas.coords(env.friend_01):
                        index += 1
                    reward, done = env.agt_reward(next_s)
                    if index > 1:
                        reward = 0

                    rewards[episode] += (cumulative_gamma * reward) / EPOCHS

                    cumulative_gamma *= GAMMA
                    next_a = RL.agt_choose(str(next_s), eps)

                    if learning is True:
                        if done or (timeStep == T - 1):
                            RL.agt_learn_final(_alpha, str(s), a, reward)
                        else:
                            RL.agt_learn_q(_alpha, str(s), a, reward, str(next_s))
                            # RL.agt_learn_sarsa(_alpha, str(s), a, reward, str(next_s), next_a)
                    s = next_s
                    a = next_a

                    # break while loop when end of this episode
                    if done or (timeStep == T - 1):
                        up_value[episode] = RL.get_q_table(str(s), 0)
                        print(up_value[episode])
                        down_value[episode] = RL.get_q_table(str(s), 1)
                        right_value[episode] = RL.get_q_table(str(s), 2)
                        left_value[episode] = RL.get_q_table(str(s), 3)
                        break

        # end of game
        print('game over')
        env.destroy()

        _index = np.zeros((EPISODES, 1))
        for i in range(EPISODES):
            _index[i] = i + 1
        _array_up = np.hstack((_index, np.reshape(up_value, (EPISODES, 1))))
        _array_down = np.hstack((_index, np.reshape(down_value, (EPISODES, 1))))
        _array_right = np.hstack((_index, np.reshape(right_value, (EPISODES, 1))))
        _array_left = np.hstack((_index, np.reshape(left_value, (EPISODES, 1))))
        mlb.plot(_array_up[:, 0], _array_up[:, 1], c='red')
        mlb.plot(_array_down[:, 0], _array_down[:, 1], c='green')
        mlb.plot(_array_right[:, 0], _array_right[:, 1], c='black')
        mlb.plot(_array_left[:, 0], _array_left[:, 1], c='blue')

        # _index = np.zeros((EPISODES, 1))
        # for i in range(EPISODES):
        #     _index[i] = i + 1
        # _array = np.hstack((_index, np.reshape(rewards, (EPISODES, 1))))
        #
        # # mlb.plot(_array[:, 0], _array[:, 1], c='red')
        # mlb.plot(_array[:, 0], _array[:, 1], c='green')
        mlb.show()


# Implement the programme
if __name__ == "__main__":
    env = Domain()
    RL = LearningAlgo(actions=list(range(env.a_length)))
    _update = UpDate()
    _epsilon = 0.1
    _alpha = 0.1
    _update.update(_epsilon, _alpha)
    # _update.update(_alpha)
    env.mainloop()


