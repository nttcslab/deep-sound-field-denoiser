import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset
import os


class SoundfieldDatasetLoader:
    """H5 file loader for sound-field dataset"""

    def __init__(self, config):
        self.dataset_dir = config["dataset_dir"]
        self.noise_type = config["noise_type"]
        self.num_sound_source = config["num_sound_source"]

        # Number of loading data from dataset file
        if config["num_total_data"] < 0:
            # Use all data in dataset
            self.num_each_data = -1
        else:
            # Number of data for each sound source
            self.num_each_data = int(
                config["num_total_data"] / len(self.num_sound_source)
            )

    def load(self) -> TensorDataset:
        """load dataset

        Returns:
            TensorDataset: sound field dataset
        """
        true_data, noisy_data = self._load_data()
        dataset = TensorDataset(noisy_data, true_data)
        return dataset

    def _load_data(self):
        true_data, noisy_data = [], []
        for num_ss in self.num_sound_source:
            tr, no = self._load_data_from_h5(num_ss)
            true_data.append(tr)
            noisy_data.append(no)

        return torch.cat(true_data, dim=0), torch.cat(noisy_data, dim=0)

    def _load_data_from_h5(self, num_ss):
        """Load sound-field dataset whose sound source number is specified by num_source.

        Args:
            num_ss (_type_): number of sound sources for loading
        Returns:
            true_data_tensor, noisy_data_tensor: true and noisy data in 4D tensor format
        """

        # load data from h5 file
        true_data_path = os.path.join(
            self.dataset_dir, f"soundsource{num_ss}", "sf_true.h5"
        )
        with h5py.File(true_data_path, "r") as f:
            if self.num_each_data > 0:
                # Load specified number of data from h5 file
                true_data = f["soundfield"][: self.num_each_data]
            else:
                # Load all data in h5 file
                true_data = f["soundfield"][:]

        noisy_data_path = os.path.join(
            self.dataset_dir, f"soundsource{num_ss}", f"sf_noise_{self.noise_type}.h5"
        )
        with h5py.File(noisy_data_path, "r") as f:
            if self.num_each_data > 0:
                noisy_data = f["soundfield"][: self.num_each_data]
            else:
                noisy_data = f["soundfield"][:]

        # normalization by true data
        true_max = np.max(np.abs(true_data), axis=(1, 2, 3))
        true_data = true_data / true_max[:, None, None, None]
        noisy_data = noisy_data / true_max[:, None, None, None]

        # cast data type
        true_data = torch.tensor(true_data).float()
        noisy_data = torch.tensor(noisy_data).float()

        return true_data, noisy_data

    def getDatasetSubset(dataset, num_data):
        subset = torch.utils.data.Subset(dataset, range(num_data))
        return subset


def main():
    from easydict import EasyDict
    import yaml

    path_to_config = "config.yml"
    assert os.path.isfile(path_to_config)
    with open(path_to_config) as f:
        yaml_contents = yaml.safe_load(f)
    config = EasyDict(yaml_contents["eval"])

    loader = SoundfieldDatasetLoader(config["dataset"])
    dataset = loader.load()
    print(len(dataset))
    print(dataset[0][0].shape)

    im_noise, im_true = dataset[:]
    print("---")
    print(im_noise.shape)


if __name__ == "__main__":
    main()
