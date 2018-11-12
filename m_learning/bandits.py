import numpy as np
# from matplotlib import pyplot as plt
from random import randint


class BanditMan:

    def __init__(self, k, time_out, bandit_name='bernoulli', p=None, adverse_p=None, p_switch=0.5):

        self.k_bandits = k
        self.time_out = time_out

        if p is None:
            self.band_p = np.absolute(np.random.rand(1, k))

        else:
            self.band_p = p

        if adverse_p is None:
            self.band_p_advs = np.absolute(np.random.rand(1, k))
        else:
            self.band_p_advs = adverse_p

        self.bandit_type = bandit_name

        self.which_bandit = {'bernoulli': self.bernoulli_bandits,
                             'advs_bernoulli': self.advs_bernoulli_bandit
                             }

        if bandit_name[0:4] == 'advs':
            flip_coin = 0
            list_p_mat = []
            for t in range(self.time_out):

                if np.remainder(t, 200) == 0:
                    flip_coin = np.random.binomial(1, p_switch)

                if flip_coin == 1:
                    list_p_mat.extend(self.band_p_advs[0:])
                else:
                    list_p_mat.extend(self.band_p[0:])

            self.p_mat = np.array(list_p_mat).T    # .reshape(self.k_bandits, self.time_out)

        else:

            list_p_mat = []
            list_p_mat.extend(list(self.band_p[0:])*self.time_out)
            self.p_mat = np.array(list_p_mat).T

        self.counter = 0

    def bernoulli_bandits(self, band_to_pull):

        p_bandit = self.band_p[0, band_to_pull]

        reward = np.random.binomial(n=1, p=p_bandit)
        return reward

    def advs_bernoulli_bandit(self, band_to_pull):

        p_bandit = self.p_mat[band_to_pull, self.counter]

        reward = np.random.binomial(n=1, p=p_bandit)

        self.counter += 1

        if self.counter == self.time_out:
            self.counter = 0

        return reward

    @staticmethod
    def cum_rewards(reward_mat):

        cum_mat = reward_mat @ np.ones((reward_mat.shape[1], 1))
        return cum_mat

    @staticmethod
    def find_mean_reward(input_mat):

        mean_vec = np.mean(input_mat, axis=1).reshape(-1, 1)
        return mean_vec

    def init_zero_mat(self):
        init_data = np.zeros((self.k_bandits, self.time_out))
        return init_data

    def count_occur(self, lst):
        na_vec = list(lst.count(x) for x in range(self.k_bandits))
        np_vec = np.array(na_vec).reshape(-1, 1)

        return np_vec

    def uniform_algo(self):

        n_explore = (2 * self.time_out * np.sqrt(2 * np.log(self.time_out))) / self.k_bandits
        n_explore = n_explore ** (2 / 3)

        uniform_band_pulls = []
        recorded_data = self.init_zero_mat()

        for t in range(self.time_out):

            if t < int(self.k_bandits * (n_explore - 2)):

                band_to_pull = int(np.remainder(t / self.k_bandits, self.k_bandits))
                uniform_band_pulls.append(band_to_pull)

                recorded_data[band_to_pull, t] = self.which_bandit[self.bandit_type](band_to_pull)
            else:

                reward_vec = self.cum_rewards(recorded_data)
                band_to_pull = np.argmax(reward_vec)
                uniform_band_pulls.append(band_to_pull)

                recorded_data[band_to_pull, t] = self.which_bandit[self.bandit_type](band_to_pull)

        return [recorded_data, uniform_band_pulls]

    def e_greedy(self):

        e_greedy_pulls = []
        recorded_data = self.init_zero_mat()

        for t in range(self.time_out):

            if t != 0:
                p_exp = (self.k_bandits * np.log(self.time_out) / self.time_out) ** (1/3)
            else:
                p_exp = 1

            explore = np.random.binomial(1, p_exp)

            if explore == 1:

                band_to_pull = randint(0, self.k_bandits-1)
                e_greedy_pulls.append(band_to_pull)

                recorded_data[band_to_pull, t] = self.which_bandit[self.bandit_type](band_to_pull)

            else:
                reward_vec = self.cum_rewards(recorded_data)
                band_to_pull = np.argmax(reward_vec)
                e_greedy_pulls.append(band_to_pull)

                recorded_data[band_to_pull, t] = self.which_bandit[self.bandit_type](band_to_pull)

        return [recorded_data, e_greedy_pulls]

    def r_a(self, mat):

        mat = mat + 0.0000001*np.ones(mat.shape)

        inter_vec = np.power(mat, -0.5).reshape(-1, 1)

        ra = np.sqrt(2*np.log(self.time_out))
        ra = ra * inter_vec
        return ra

    def ucb_one(self):

        ucb_pulls = []
        recorded_data = self.init_zero_mat()

        band_to_pull = randint(0, self.k_bandits - 1)
        ucb_pulls.append(band_to_pull)

        recorded_data[band_to_pull, 1] = self.which_bandit[self.bandit_type](band_to_pull)

        for t in range(1, self.time_out):

            lor_loit_vec = self.find_mean_reward(recorded_data[:, 0:t]) + self.r_a(self.count_occur(ucb_pulls))

            band_to_pull = np.argmax(lor_loit_vec)
            ucb_pulls.append(band_to_pull)

            recorded_data[band_to_pull, t] = self.which_bandit[self.bandit_type](band_to_pull)

        return [recorded_data, ucb_pulls]

    def regret_at_t(self, reward_mat, t):

        mean_vec = self.find_mean_reward(reward_mat[:, 0:t])

        mu_star = np.amax(mean_vec, axis=0)
        cum_mu_arms = np.sum(mean_vec, axis=0)

        regret = mu_star * t - cum_mu_arms
        return regret

    def regret_with_mean(self, reward_lst, t, opt_p=True):

        pulled_bands = reward_lst[1]
        p_at_t = self.p_mat[:, 0:t]

        max_p = np.amax(p_at_t, axis=0)

        mean_sum = 0
        if opt_p is True:
            for t in range(len(pulled_bands[0:t])):

                mean_sum += p_at_t[pulled_bands[t], t]
        elif opt_p is False:

            mean_sum = np.sum(np.sum(reward_lst[0], axis=1)/self.time_out, axis=0)

        regret = np.sum(max_p) - mean_sum
        return regret

    def thompsamp(self, alpha=1, beta=1):

        success_arms = np.zeros((self.k_bandits, 1))
        fail_arms = np.zeros((self.k_bandits, 1))
        theta_arms = np.empty((self.k_bandits, 1))

        thomp_pulls = []
        recorded_data = self.init_zero_mat()

        for t in range(self.time_out):
            for i in range(self.k_bandits):

                theta_arms[i, 0] = np.random.beta(success_arms[i, 0]+alpha, fail_arms[i, 0]+beta)

            band_to_pull = np.argmax(theta_arms, axis=0)

            thomp_pulls.append(band_to_pull)

            reward_at_t = self.which_bandit[self.bandit_type](band_to_pull)
            recorded_data[band_to_pull, t] = reward_at_t

            if reward_at_t == 1:
                success_arms[band_to_pull, 0] += 1
            else:
                fail_arms[band_to_pull, 0] += 1

        return [recorded_data, thomp_pulls]

    def algo_reward(self, reward_mat):

        total_reward = np.sum(self.cum_rewards(reward_mat), axis=0)
        return total_reward


if __name__ == "__main__":

    stock_bands = BanditMan(k=10, time_out=1000, bandit_name='bernoulli')
    """ Run the simulations with the different solving algorithms """
    uniform_sim = stock_bands.uniform_algo()

    e_greedy_sim = stock_bands.e_greedy()

    ucb_sim = stock_bands.ucb_one()

    thomp_sim = stock_bands.thompsamp()
    """ Begin Analysis of the rewards """

    cumlative_reward = stock_bands.cum_rewards(uniform_sim[0])

    [uni_reg, e_reg, ucb_reg] = [stock_bands.regret_at_t(uniform_sim[0], stock_bands.time_out), stock_bands.regret_at_t(
                                    e_greedy_sim[0], stock_bands.time_out),
                                 stock_bands.regret_at_t(ucb_sim[0], stock_bands.time_out)]

    thomp_sim_regret = stock_bands.regret_with_mean(thomp_sim, stock_bands.time_out)
    """"
    fig1 = plt.figure()

    plt.step(np.linspace(1, 10, 10), cumlative_reward)
    plt.step(np.linspace(1, 10, 10), stock_bands.cum_rewards(e_greedy_sim[0]))
    plt.step(np.linspace(1, 10, 10), stock_bands.cum_rewards(ucb_sim[0]))
    plt.step(np.linspace(1, 10, 10), stock_bands.cum_rewards(thomp_sim[0]))

    
    fig2 = plt.figure()

    plt.hist(uniform_sim[1])
    
    plt.hist(e_greedy_sim[1])
    plt.hist(ucb_sim[1])
    plt.hist(thomp_sim[1])
    
    plt.show()
    """
