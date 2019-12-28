
import numpy as np


class KArmedBandit:

    def __init__(self, k_arms):
        self.time_step = 0
        self.reward = 0
        self.action_selected = 0
        self.k_arms = k_arms
        self.reward_distributions = self.create_reward_distributions()

    # Returns a dict containing    action_number:{[rewards,reward_distributions]}
    # Input: Number of K_arms (1 - 10)
    def create_reward_distributions(self):

        actual_estimates = []
        reward_distribution = {}
        prob_dist = self.create_probability_distribution()
        rewards = self.create_random_rewards(prob_dist)

        for i in range(len(prob_dist)):
            reward_distribution.update({i: [rewards[i], prob_dist[i]]})

        print("Reward distribution: ", reward_distribution)

        ############## CALCULATE WHICH ACTION IS THE BEST ##############

        for key in range(len(reward_distribution)):
            current = 0
            for item in range(len(reward_distribution[key][0])):
                current += reward_distribution[key][0][item] * reward_distribution[key][1][item]
            actual_estimates.append(current)
        # q(1) = (.5 * 11) + (.5 * 9)
        print("Actual estimates: ", actual_estimates, " Best action: ", np.argmax(actual_estimates))
        ###################################################################


        return reward_distribution


    # Takes in a list of prop dist and returns a list of random reward for each prop
    def create_random_rewards(self, prop_dis):
        rewards = []

        for i in range(len(prop_dis)):
            temp_reward = []
            for x in range(len(prop_dis[i])):
                temp_reward.append(int(np.random.choice(np.arange(1, 21), 1)))  # Play around with different ranges

            rewards.append(temp_reward)
        return rewards

    # Assigns a random number of rewards for each arm and then comes up with a random probability of getting that reward
    # Returns a list of lists containing the prop for each arm [[0.9,0.1], [0.4,0.5], [0.2,0.2,0.4,0.1]]
    def create_probability_distribution(self):

        distributions = []

        for arm in range(self.k_arms):
            probability_distributions = []
            number_of_rewards = int(np.random.choice(np.arange(2, 5), 1))  # Generates a random number of reward per arm
            max_prop = 1.0 - ((number_of_rewards - 1) / 10)  # Determines the maximum probability for the first reward making sure to leave 10% for the rest as a minimum
            total_prop = 0

            for reward_number in range(number_of_rewards):

                if reward_number != number_of_rewards - 1:

                    # Creates a range of distribution values and randomly selects one
                    available_prop = self.create_range(0.1, max_prop)
                    selected_prop = np.random.choice(list(available_prop), 1)
                    total_prop += selected_prop

                    number_of_remaining_rewards = ((number_of_rewards - (reward_number + 2)) / 10)  #
                    max_prop = (1.0 - total_prop) - number_of_remaining_rewards  # calculates a new max_prop based on how many rewards are left
                    probability_distributions.append(selected_prop[0])

                else:
                    selected_prop = max_prop
                    probability_distributions.append(np.around(float(selected_prop), 1))  # Fix to many decimal when doing this way

            distributions.append(probability_distributions)

        return distributions


    # Returns list of specified range with step size of 0.1 (0.1, 0.4) -> 0.1 0.2 0.3 0.4 both params inclusive
    def create_range(self, start, end):
        tally = start
        prop_range = []

        for i in range(int(end * 10)):
            prop_range.append(np.around(tally, 1))
            tally += 0.1

        # Fix for error occurring when start: 0.1 & end:0.1 returning an empty list. possibly due to rounding 0.1 to 0
        if len(prop_range) != 0:
            return prop_range
        else:
            prop_range = [0.1]
            return prop_range


    def get_reward(self, arm):

        reward = np.random.choice(self.reward_distributions[arm][0], 1, p=self.reward_distributions[arm][1])

        return int(reward)


