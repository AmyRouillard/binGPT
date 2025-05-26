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
        ind_least_significant = 0
        for i in range(length):
            if np.random.rand() > 0.5:
                x += "1"
                ind_least_significant = i
            else:
                x += "0"

        x_tmp = [x]
        for _ in range(self.iterations):
            flip = False if x_tmp[-1][0] == "0" else True
            if flip:
                y = ""
                for t in x_tmp[-1][1:ind_least_significant]:
                    y += "1" if t == "0" else "0"
            else:
                y = x_tmp[-1][1:ind_least_significant]

            y += x_tmp[-1][ind_least_significant:]
            x_tmp.append(y)
            if ind_least_significant > 1:
                ind_least_significant -= 1

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


class TentDataset(Dataset):
    """ """

    def __init__(
        self,
        split,
        length=6,
        n_iterations=1,
        type="binary",
        tokenized=True,
        in_test=None,
    ):
        assert split in {"train", "test", "validation"}
        assert type in {"binary", "decimal"}
        assert length > 2

        self.split = split
        self._length = length
        self.n_iterations = n_iterations
        self.type = type
        self.tokenized = tokenized

        if type == "binary":
            self.length = self._length
        elif type == "decimal":
            self.length = 1
            while (10 ** (self.length) * 2 ** (-self._length)) % 1 != 0:
                self.length += 1
        else:
            raise ValueError("type must be either 'binary' or 'decimal'")

        if in_test is None:
            # 2**-n of the binary digits are in test set
            n = 4
            in_test = list("1" * (2 ** (self._length - n))) + list(
                "0" * (2 ** (self._length - n) * (2**n - 1))
            )

            # shuffle the in_test list with a fixed seed
            rng = np.random.default_rng(42)
            in_test = rng.permutation(in_test).tolist()

        if self.split == "test":
            self.map_idx = [i for i in range(len(in_test)) if in_test[i] == "1"]
        elif self.split == "train":
            self.map_idx = [i for i in range(len(in_test)) if in_test[i] == "0"]
        elif self.split == "validation":
            self.map_idx = [i for i in range(len(in_test)) if in_test[i] == "2"]

        # self.token_map = {
        #     "0": 0,
        #     "1": 1,
        #     "$": 2,  # eos
        #     ">": 3,  # separator
        #     " ": 4,  # pad
        # }
        self.vocab_size = 2 if type == "binary" else 10

        # assert len(self.map_idx) == self.__len__()

    def __len__(self):
        return len(self.map_idx)

    def get_vocab_size(self):
        if self.tokenized:
            return self.vocab_size
        else:
            return None

    def generate_data_sequence(self, idx=None):

        if idx is None:
            # generate some random integers from [0, self.__len__())
            x = np.random.randint(0, self.__len__())
        else:
            x = idx

        # convert x into a binary string
        x = bin(x)[2:].zfill(self._length)

        ind_least_significant = x.rfind("1")

        x_tmp = [x]
        for _ in range(self.n_iterations):

            if ind_least_significant == -1:
                # if ind_least_significant == -1, then x is zero, which remains zero
                y = "0" * self._length
            elif ind_least_significant == 0:
                # if ind_least_significant == 0, then x is 1/2, which becomes 1
                y = "1" * self._length
                ind_least_significant = -1
            else:
                # if ind_least_significant > 0 then follow the usual rule
                flip = False if x_tmp[-1][0] == "0" else True
                if flip:
                    y = ""
                    for t in x_tmp[-1][1:ind_least_significant]:
                        y += "1" if t == "0" else "0"
                else:
                    y = x_tmp[-1][1:ind_least_significant]

                y += x_tmp[-1][ind_least_significant:]
                y += "0"  # pad y with a zero
                ind_least_significant -= 1

            x_tmp.append(y)

        x0 = x_tmp[0]
        x1 = x_tmp[-1]

        x0 = [int(d) for d in x0]
        x1 = [int(d) for d in x1]

        if self.type == "decimal":

            x0 = sum([d / 2**i / 2 for i, d in enumerate(x0)])
            x1 = sum([d / 2**i / 2 for i, d in enumerate(x1)])

            if self.tokenized:
                x0 = format(x0, f".{self.length}f")[2:]
                x1 = format(x1, f".{self.length}f")[2:]
                x0 = [int(d) for d in x0]
                x1 = [int(d) for d in x1]
            else:
                x0 = [x0]
                x1 = [x1]

        # convert to torch tensors
        return (
            torch.tensor(x0),  # , dtype=torch.long
            torch.tensor(x1),  # , dtype=torch.long
        )

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input (self._length)
        # and the output (self._length - self.n_iterations), but -1 because
        # the transformer starts making predictions at the last input element
        return self._length * 2 - 1  # - self.n_iterations

    def __getitem__(self, idx):

        inp, sol = self.generate_data_sequence(self.map_idx[idx])

        if self.tokenized:
            # concatenate the problem specification and the solution
            cat = torch.cat((inp, sol), dim=0)

            # the inputs to the transformer will be the offset sequence
            x = cat[:-1].clone()
            y = cat[1:].clone()
            # we only want to predict at output locations, mask out the loss at the input locations
            y[: self._length - 1] = -1

            # assert x and y have length self.get_block_size()
            assert x.size(0) == self.get_block_size()
            assert y.size(0) == self.get_block_size()

            return x, y
        else:
            return inp, sol


# class ProbeDataset(TentDataset):
#     """flip or no-flip: check if the nth bit of batch is 1 or 0"""

#     def __init__(self, split, length=6, n_iterations=1, type="binary"):
#         super().__init__(split, length, n_iterations, type)
#         assert split in {"train", "test"}

#         self.n_classes = 2  # flip or no-flip

#     def __getitem__(self, idx):

#         inp, sol = self.generate_data_sequence(self.map_idx[idx])
#         cat = torch.cat((inp, sol), dim=0)

#         # the inputs to the transformer will be the offset sequence
#         x = cat[:-1].clone()
#         y = (x[self.n_iterations] == 1).long()

#         # assert x and y have length self.get_block_size()
#         assert x.size(0) == self.get_block_size()

#         return x, y


class ProbeDataset(TentDataset):
    """find the position of the least significant bit in the input sequence"""

    def __init__(self, split, length=6, n_iterations=1, type="binary"):
        super().__init__(split, length, n_iterations, type)
        assert split in {"train", "test"}

        self.n_classes = length  # find the position of the least significant bit

    def __getitem__(self, idx):

        inp, sol = self.generate_data_sequence(self.map_idx[idx])
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        # where is x ==1
        y = (inp == 1).nonzero()[-1].long()

        # assert x and y have length self.get_block_size()
        assert x.size(0) == self.get_block_size()

        return x, y
