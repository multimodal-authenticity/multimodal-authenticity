import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MMDataset(Dataset):
    """Multimodal Dataset Loader.
    Args:
        args (dict): Configuration dictionary containing:
            - dataset_name: Name of the dataset to load
            - featurePath: Path to the preprocessed feature file
            - feature_dims: List to store feature dimensions [text, audio, vision]
            - need_normalized: Boolean flag for feature normalization
    """

    def __init__(self, args):
        """Initialize dataset based on configuration."""
        self.args = args

        # Mapping of dataset names to their initialization methods
        DATASET_MAP = {
            'Authenticity': self.__init_evaluation,  # Evaluation dataset for authenticity detection
        }

        # Call the appropriate initialization method
        DATASET_MAP[args['dataset_name']]()

    def __init_evaluation(self):
        """Initialize evaluation dataset from preprocessed features."""
        # Load preprocessed data from pickle file
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)

        # Extract and convert modalities to float32
        self.text = data['text'].astype(np.float32)  # Text features [N, seq_len, dim]
        self.vision = data['vision'][:, :, :].astype(np.float32)  # Visual features [N, seq_len, dim]
        self.audio = data['audio'].astype(np.float32)  # Audio features [N, seq_len, dim]
        self.meta_info = data['info']  #  infomation
        # Process labels (subtracting 4 for normalization/centering)#  amend [1,7] to [-3,3]
        self.labels = { 'M': np.array(data['labels'] - 4).astype(np.float32)   }

        # Handle missing values (NaN) by zero-imputation
        self.vision[self.vision != self.vision] = 0  # vision != vision detects NaN
        self.audio[self.audio != self.audio] = 0
        self.text[self.text != self.text] = 0

        # Update feature dimensions in configuration
        self.args['feature_dims'] = [
            self.text.shape[2],  # Text feature dimension
            self.audio.shape[2],  # Audio feature dimension
            self.vision.shape[2]  # Visual feature dimension
        ]

        # Apply normalization if configured
        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()


    def __normalize(self):
        """
        Normalize the audio and vision features by taking the mean across examples.
        This reduces the variability and removes potential NaN values.
        """
        # Transpose (num_examples, max_len, feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

        # Compute the mean over the sequence length (max_len)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # Handle any remaining NaN values by setting them to 0
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        # Transpose back to the original shape (num_examples, max_len, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        """
        return self.text.shape[0]

    def get_seq_len(self):
        """
        Get the sequence lengths for text, audio, and vision modalities.

        :return: A tuple containing sequence lengths for text, audio, and vision.
        """
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def __getitem__(self, index):
        """
        Get a single sample from the dataset at the specified index.

        :param index: Index of the sample to retrieve.
        :return: A dictionary containing text, audio, vision features, labels, and meta information.
        """
        # Create the sample dictionary with text, audio, vision, and labels
        sample = {
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            "meta_info": self.meta_info[index]
        }

        # Add sequence lengths for audio and vision
        sample['audio_lengths'] = self.audio.shape[0]
        sample['vision_lengths'] = self.vision.shape[0]

        return sample


# DataLoader function for handling multiple dataset splits (train, valid, test)
def MMDataLoader(args, num_workers=0):
    # Create datasets
    datasets =   MMDataset(args)
    # Get the sequence lengths for each modality if specified in the arguments
    args['seq_lens'] = datasets.get_seq_len()
    # Create DataLoader objects for each split
    dataLoader =  DataLoader(datasets,
                       batch_size=128,
                       num_workers=num_workers,
                       shuffle=False)
    return dataLoader
