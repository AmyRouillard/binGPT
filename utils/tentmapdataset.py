import numpy as np


class TentMap2Dataset:
    def __init__(self, distribution="uniform", max_length=10, iterations=1):
        if distribution in ["fixed", "uniform"]:
            self.distribution = distribution
        else:
            raise ValueError("distribution must be either 'fixed' or 'uniform'")

        self.iterations = iterations

        self.min_length = self.iterations + 1
        self.max_length = max_length

        if self.iterations >= self.max_length:
            raise ValueError("iterations must be less than the max_length")

    def generate_data_pair(self, length):
        x = ""
        ind_most_significant = 0
        for i in range(length):
            if np.random.rand() > 0.5:
                x += "1"
                ind_most_significant = i
            else:
                x += "0"

        x_tmp = [x]
        for _ in range(self.iterations):
            flip = False if x_tmp[-1][0] == "0" else True
            if flip:
                y = ""
                for t in x_tmp[-1][1:ind_most_significant]:
                    y += "1" if t == "0" else "0"
            else:
                y = x_tmp[-1][1:ind_most_significant]

            y += x_tmp[-1][ind_most_significant:]
            x_tmp.append(y)
            if ind_most_significant > 1:
                ind_most_significant -= 1

        return x_tmp

    def generate_batch(self, N):
        X, Y = [], []
        if self.distribution == "fixed":
            for _ in range(N):
                x = self.generate_data_pair(self.max_length)
                X.append(x[0])
                Y.append(x[-1])
            return X, Y
        else:
            for _ in range(N):
                x = self.generate_data_pair(
                    np.random.randint(self.min_length, self.max_length + 1)
                )
                X.append(x[0])
                Y.append(x[-1])
            return X, Y
