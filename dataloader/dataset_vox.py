import os, glob, random
import json

import torch
import torchaudio


class DatasetVoxCeleb2(torch.utils.data.Dataset):
    def __init__(self, data_path="/workspace/db/audio/vox2/dev/aac/"):
        self.data_path = data_path
        self.speaker_ids = sorted(os.listdir(self.data_path))
        try:
            with open('./dataloader/anno.json') as json_file:
                self.anno = json.load(json_file)
        except FileNotFoundError:
            self._get_anno_()
        self.utterance_num = 10
        self.speaker_num = 64
        
    def _get_anno_(self):
        anno = list()
        speaker_ids = sorted(os.listdir(self.data_path))
        for id, id_name in enumerate(speaker_ids):
            print(id)
            speaker_anno = {
                "id" : id,
                "file_names" : glob.glob(os.path.join(self.data_path, id_name + "/*/*.wav"))
            }
            anno.append(speaker_anno)
        with open('dataloader/anno.json', 'w') as fout:
            json.dump(anno , fout)

    def _get_samples_(self):
        speaker_list = random.sample(self.anno, self.speaker_num)
        speaker_utterance_list = list(map(lambda item:random.sample(item["file_names"], self.utterance_num), speaker_list))
        return speaker_utterance_list

    def _VoiceActivityDetector_(self):
        i = 0
        for speaker in self.anno:
            i += 1
            for utterance in speaker["file_names"]:
                wav_tensor, sr = torchaudio.load(utterance)
                wav_tensor = torchaudio.functional.vad(wav_tensor, sr)
                torchaudio.save(utterance[:-4] + "VAD.wav", wav_tensor, sr)
                progress = 100*i/len(self.anno)
                print("Progress: %0.3f" % progress + chr(37), utterance[:-4] + "VAD.wav")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        speaker_utterance_list = self._get_samples_()
        self._VoiceActivityDetector_(speaker_utterance_list)
        # out, sr = torchaudio.load(self.anno[idx]["file_names"][0])
        raise NotImplementedError


if __name__ == "__main__":
    dataset = DatasetVoxCeleb2()
    dataset._VoiceActivityDetector_()
    # dataset._get_anno_()