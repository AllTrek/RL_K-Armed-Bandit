from k_arm_env import KArmedBandit
import numpy as np
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, arms):
        self.arms = arms
        self.init_env = KArmedBandit(arms)
        self.track_values = self.set_dict()

    def set_dict(self):
        default_dict = {}
        for i in range(self.arms):
            default_dict.update({i: [0, 0]})

        return default_dict

    # Dict containing the previous estimate and a tally of how many times that action has been called
    def update_estimate_dict(self, arm, current_reward):

        # Update estimate | use previous value of time? Try out by hand normal method vs incremental
        self.update_estimate(arm, current_reward)

        # Update the number of times called
        previous_tally = self.track_values[arm][1]
        self.track_values[arm][1] = previous_tally + 1

    def update_estimate(self, arm, current_reward):

        previous_estimate = self.track_values[arm][0]


        tally = self.track_values[arm][1]

        if tally == 0:
            tally = 1

        new_estimate = previous_estimate + 1/tally * (current_reward - previous_estimate)


        self.track_values[arm][0] = new_estimate

    def get_max_estimate(self):
        highest = float('-inf')
        index = 0

        for i in range(len(self.track_values)):

            if self.track_values[i][0] > highest:
                highest = self.track_values[i][0]
                index = i

        return index



agent = Agent(4)

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.00  # 0.01
decay_rate = 0.001  # 0.01 works faster but 0.001 with 10000 more accurate

rewards_over_time = []
X = []
action_over_time = []


for timestep in range(10000):

    # Epsilon greedy to select arm
    random = np.random.uniform(0, 1)

    if random > epsilon:
        action = agent.get_max_estimate()
    else:
        action = np.random.randint(1, (len(agent.track_values) + 1))
        action -= 1



    reward = agent.init_env.get_reward(action)

    # Update the values
    agent.update_estimate_dict(action, reward)
    rewards_over_time.append(reward)
    action_over_time.append(action)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * timestep)




print("Predicated estimates: ", agent.track_values)


for i in range(len(action_over_time)):
    X.append(i)


plt.scatter(X, action_over_time)
plt.show()

