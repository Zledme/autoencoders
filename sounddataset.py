import torch
from torch.utils.data import Dataset
import os
import torchaudio

class transformed_Dataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.files = os.listdir(root)


    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):

        file_name = self.files[idx]

        file_path = os.path.join(self.root, file_name)

        waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        if self.transforms:
            waveform = self.transforms(waveform)

        log_offset = 1e-6
        log_mel_specgram = torch.log(waveform + log_offset)


        return log_mel_specgram, file_name


