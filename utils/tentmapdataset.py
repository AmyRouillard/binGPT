import numpy as np
from torch.utils.data import Dataset
import torch
import pickle


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

        print("This class is deprecated. Use SortDataset instead.")

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


class SortDataset(Dataset):
    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, n_iterations=1):
        assert split in {"train", "test"}
        assert length > 2
        self.split = split
        self.length = length
        self.n_iterations = n_iterations

        in_test = list("1" * (2 ** (self.length - 2))) + list(
            "0" * (2 ** (self.length - 2) * 3)
        )
        # shuffle the in_test list with a fixed seed
        np.random.seed(42)
        np.random.shuffle(in_test)

        if self.split == "test":
            self.map_idx = [i for i in range(len(in_test)) if in_test[i] == "1"]
        else:
            self.map_idx = [i for i in range(len(in_test)) if in_test[i] == "0"]

        # self.token_map = {
        #     "0": 0,
        #     "1": 1,
        #     "$": 2,  # eos
        #     ">": 3,  # separator
        #     " ": 4,  # pad
        # }
        self.vocab_size = 2  # binary

        assert len(self.map_idx) == self.__len__()

    def __len__(self):
        if self.split == "train":
            return 2 ** (self.length - 2) * 3
        else:
            return 2 ** (self.length - 2)

    def get_vocab_size(self):
        return self.vocab_size

    def generate_data_sequence(self, idx=None):

        if idx is None:
            # generate some random integers from [0, self.__len__())
            x = np.random.randint(0, self.__len__())
        else:
            x = idx

        # convert x into a binary string
        x = bin(x)[2:].zfill(self.length)

        ind_most_significant = x.rfind("1")
        # x = ""
        # ind_most_significant = -1
        # for i in range(self.length):
        #     if np.random.rand() > 0.5:
        #         x += "1"
        #         ind_most_significant = i
        #     else:
        #         x += "0"

        x_tmp = [x]
        for _ in range(self.n_iterations):

            if ind_most_significant == -1:
                # if ind_most_significant == -1, then x is zero, which remains zero
                y = "0" * self.length
            elif ind_most_significant == 0:
                # if ind_most_significant == 0, then x is 1/2, which becomes 1
                y = "1" * self.length
                ind_most_significant = -1
            else:
                # if ind_most_significant > 0 then follow the usual rule
                flip = False if x_tmp[-1][0] == "0" else True
                if flip:
                    y = ""
                    for t in x_tmp[-1][1:ind_most_significant]:
                        y += "1" if t == "0" else "0"
                else:
                    y = x_tmp[-1][1:ind_most_significant]

                y += x_tmp[-1][ind_most_significant:]
                y += "0"  # pad y with a zero
                ind_most_significant -= 1

            x_tmp.append(y)

        # return x_tmp

        # convert to torch tensors
        return (
            torch.tensor([int(d) for d in x_tmp[0]], dtype=torch.long),
            torch.tensor([int(d) for d in x_tmp[-1]], dtype=torch.long),
        )

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input (self.length)
        # and the output (self.length - self.n_iterations), but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1  # - self.n_iterations

    def __getitem__(self, idx):

        # # use rejection sampling to generate an input example from the desired split
        # while True:
        #     # # generate some random integers
        #     # inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
        #     # # half of the time let's try to boost the number of examples that
        #     # # have a large number of repeats, as this is what the model seems to struggle
        #     # # with later in training, and they are kind of rate
        #     # if torch.rand(1).item() < 0.5:
        #     #     if inp.unique().nelement() > self.length // 2:
        #     #         # too many unqiue digits, re-sample
        #     #         continue
        #     inp, sol = self.generate_data_sequence()

        #     # figure out if this generated example is train or test based on its hash
        #     h = hash(pickle.dumps(inp.tolist()))
        #     inp_split = (
        #         "test" if h % 4 == 0 else "train"
        #     )  # designate 25% of examples as test
        #     if inp_split == self.split:
        #         break  # ok

        inp, sol = self.generate_data_sequence(self.map_idx[idx])
        # solve the task: i.e. sort
        # sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[: self.length - 1] = -1

        # assert x and y have length self.get_block_size()
        assert x.size(0) == self.get_block_size()
        assert y.size(0) == self.get_block_size()

        return x, y
