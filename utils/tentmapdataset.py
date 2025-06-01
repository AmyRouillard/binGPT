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

        self.vocab_size = 2 if type == "binary" else 10

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
        return self._length

    def __getitem__(self, idx):

        inp, sol = self.generate_data_sequence(self.map_idx[idx])

        if self.tokenized:
            assert inp.size(0) == self.get_block_size()
            assert sol.size(0) == self.get_block_size()

            return inp, sol
        else:
            return inp, sol


class ProbeDataset(TentDataset):
    """find the position of the least significant bit in the input sequence"""

    def __init__(
        self,
        split,
        length=6,
        n_iterations=1,
        type="binary",
        tokenized=True,
        in_test=None,
    ):
        super().__init__(
            split,
            length,
            n_iterations,
            "binary",
            tokenized=tokenized,
            in_test=in_test,
        )

        self.ptype = type
        self.n_classes = (
            length + 1
        )  # find the position of the least significant bit (number of bits + 1 for no bit set)

    def __getitem__(self, idx):

        inp, _ = self.generate_data_sequence(self.map_idx[idx])

        # where is x ==1
        y = (inp == 1).nonzero()
        if y.size(0) > 0:
            y = y[-1].long()
        else:
            y = torch.tensor([self.length], dtype=torch.long)

        if self.ptype == "decimal":
            x0 = inp.tolist()
            x0 = sum([d / 2**i / 2 for i, d in enumerate(x0)])

            if self.tokenized:
                x0 = format(x0, f".{self.length}f")[2:]
                x0 = [int(d) for d in x0]
            else:
                x0 = [x0]

            inp = torch.tensor(x0)

        return inp, y


class ProbeDatasetMod(TentDataset):
    """find the position of the least significant bit in the input sequence"""

    def __init__(
        self,
        split,
        length=6,
        n_iterations=1,
        type="binary",
        tokenized=True,
        in_test=None,
        target_step=1,
    ):
        super().__init__(
            split,
            length,
            n_iterations,
            "binary",
            tokenized=tokenized,
            in_test=in_test,
        )
        self.ptype = type
        self.n_classes = (
            length + 1
        )  # find the position of the least significant bit (number of bits + 1 for no bit set)

        self.target_step = target_step
        self.steps = [i for i in range(-target_step, target_step + 1) if i != 0]

    def generate_modified_data_sequence(self, idx=None):

        if idx is None:
            # generate some random integers from [0, self.__len__())
            x = np.random.randint(0, self.__len__())
        else:
            x = idx

        # convert x into a binary string
        x = bin(x)[2:].zfill(self._length)

        ind_least_significant = x.rfind("1")
        old_ind_least_significant = x.rfind("1")
        ##################
        # randomly select a step to modify the least significant bit
        # if ind_least_significant == 0:
        #     # target_modification = torch.randint(
        #     #     low=len(self.steps) // 2, high=len(self.steps), size=1, dtype=torch.long
        #     # )
        #     idx_mod = np.random.randint(
        #         low=len(self.steps) // 2,
        #         high=len(self.steps),
        #     )
        # elif ind_least_significant == self.length - 1:
        #     # idx_mod = torch.randint(
        #     #     low=0, high=len(self.steps) // 2, size=1, dtype=torch.long
        #     # )
        #     idx_mod = np.random.randint(
        #         low=0,
        #         high=len(self.steps) // 2,
        #     )
        # else:
        #     # idx_mod = torch.randint(
        #     #     low=0, high=len(self.steps) // 2, size=1, dtype=torch.long
        #     # )
        #     idx_mod = np.random.randint(
        #         low=0,
        #         high=len(self.steps),
        #     )
        # ind_least_significant += self.steps[idx_mod]

        # reduce the index by the target step
        ind_least_significant -= self.target_step

        # clamp the index to be within the bounds of the string
        ind_least_significant = max(-1, min(ind_least_significant, self._length - 1))
        x0 = [ind_least_significant if ind_least_significant >= 0 else self.length]

        ####################
        # if ind_least_significant >= 0:
        #     # make sure the least significant bit is set to 1
        #     x = x[:ind_least_significant] + "1" + x[ind_least_significant + 1 :]

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

                # TODO: ignore later bits - set to zero? or keep them?
                # keep the bits after the least significant bit
                y += x_tmp[-1][ind_least_significant:]
                y += "0"  # pad y with a zero
                # ignore bits after the least significant bit
                # y += "0" * (len(x_tmp[-1][ind_least_significant:]) + 1)
                ###

                ind_least_significant -= 1

            x_tmp.append(y)

        # x0 = x_tmp[0]
        x1 = x_tmp[-1]

        # x0 = [int(d) for d in x0]
        x1 = [int(d) for d in x1]

        return (
            torch.tensor(x0),  # , dtype=torch.long
            torch.tensor(x1),  # , dtype=torch.long
        )

    def __getitem__(self, idx):

        inp, out = self.generate_data_sequence(self.map_idx[idx])
        y_mod, out_mod = self.generate_modified_data_sequence(self.map_idx[idx])

        # where is x ==1
        y = (inp == 1).nonzero()
        if y.size(0) > 0:
            y = y[-1].long()
        else:
            y = torch.tensor([self.length], dtype=torch.long)

        if self.ptype == "decimal":
            x0 = inp.tolist()
            x1 = out.tolist()
            x0 = sum([d / 2**i / 2 for i, d in enumerate(x0)])
            x1 = sum([d / 2**i / 2 for i, d in enumerate(x1)])

            out_mod = out_mod.tolist()
            out_mod = sum([d / 2**i / 2 for i, d in enumerate(out_mod)])
            if self.tokenized:
                x0 = format(x0, f".{self.length}f")[2:]
                x1 = format(x1, f".{self.length}f")[2:]

                out_mod = format(out_mod, f".{self.length}f")[2:]

                x0 = [int(d) for d in x0]
                x1 = [int(d) for d in x1]

                out_mod = [int(d) for d in out_mod]
            else:
                x0 = [x0]
                x1 = [x1]
                out_mod = [out_mod]

            inp = torch.tensor(x0)
            out = torch.tensor(x1)
            out_mod = torch.tensor(out_mod)

        return inp, y, y_mod, out, out_mod
