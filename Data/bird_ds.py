import os
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torchvision.transforms as transforms

class BirdsDS(Dataset):
    def __init__(self, root_path, phase='train'): # train val test all
        self.root_path = root_path
        self.data_dir = os.path.dirname(os.path.dirname(self.root_path)) + '/3areas-10birds-v02/data_wav_8s'
        self.ds_txt = os.path.join(self.root_path, phase + '_set.txt')
        with open(self.ds_txt) as f:
            files = f.readlines()
        self.files = [line.strip() for line in files]

        self.mel_spec_extractor = nn.Sequential(MelSpectrogram(16000, n_fft=2048, hop_length=512, n_mels=128),
                                                AmplitudeToDB())
        
        self.class_dict = {
                            'agelaius_phoeniceus': 0,
                            'molothrus_ater': 1,
                            'tringa_semipalmata': 2,
                            'cardinalis_cardinalis': 3,
                            'setophaga_aestiva': 4,
                            'turdus_migratorius': 5,
                            'certhia_americana': 6,
                            'setophaga_ruticilla': 7,
                            'corvus_brachyrhynchos': 8,
                            'spinus_tristis': 9
                        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = os.path.join(self.data_dir, self.files[idx])
        
        waveform, sample_rate = torchaudio.load(file)
        spec = self.mel_spec_extractor(waveform)

        label_txt = file.split('/')[-2].lower()
        label = self.class_dict[label_txt]
        
        return spec, label, label_txt


class BirdsDS_IMG(BirdsDS):
    def __init__(self, root_path, phase='train'):
        super().__init__(root_path, phase)
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        spec, label, label_txt = super().__getitem__(idx)
        spec -= spec.min()
        spec /= spec.max()
        # spec = self.to_pil(spec)
        # spec = self.to_tensor(spec).squeeze(0)
        return spec.squeeze(0), label


if __name__ == '__main__':
    root_path = '/nas/staff/data_work/Xin/birds-xie/meta/1'
    audio_dataset = BirdsDS(root_path)

    # Accessing the first item in the dataset
    waveform, sample_rate = audio_dataset[0]

    print(f"Waveform shape: {waveform.shape}")
    print(f"Sample rate: {sample_rate}")
