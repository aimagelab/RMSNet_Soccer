import torch
import torch.utils.data as data_utl
import numpy as np
import os
import random
from data.load_data_utils import FPS, LABELS, load_events, load_background_train, sample_training, sample_background, load_background_val, augment_training
from PIL import Image
from torchvision import transforms

np.random.seed(123)
torch.manual_seed(123)
random.seed(123)

def sort_function(event):
    return (event["match"], event["half"], event["frame_indexes"][0])

class SoccerNet(data_utl.Dataset):

    def __init__(self, frames_per_clip=41, resize_to=(224, 398), split="train", frames_path="", labels_path="", listgame_path="", class_samples_per_epoch=1000, test_overlap=0):
        self.split = split
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.listgame_file = []
        self.resize_to = resize_to

        if "+" in self.split: #we want to train on more than 1 split (eg train+val)
            splits = self.split.split("+")
            for s in splits:
                self.listgame_file.append(os.path.join(listgame_path, "listgame_"+s+".npy"))
        else:
            self.listgame_file.append(os.path.join(listgame_path, "listgame_"+self.split+".npy"))
        self.class_samples_per_epoch = class_samples_per_epoch
        self.test_overlap = test_overlap
        self.frames_per_clip = frames_per_clip
        self.matches = np.load(self.listgame_file[0])
        if len(self.listgame_file)>1:
            for lf in self.listgame_file[1:]:
                self.matches = np.concatenate((self.matches, np.load(lf)))
        self.matches.sort()
        self.n_frames_per_half_match = {}
        for match in self.matches:
            self.n_frames_per_half_match[match + "_1"] = len([name for name in os.listdir(os.path.join(frames_path, match.replace("/", "_") + "_1"))])
            self.n_frames_per_half_match[match + "_2"] = len([name for name in os.listdir(os.path.join(frames_path, match.replace("/", "_") + "_2"))])

        if "train" in self.split:
            self.interesting_events = load_events(self.matches, self.labels_path, self.frames_per_clip)
            self.augmented_events = augment_training(self.interesting_events, self.frames_per_clip)
            self.all_background_events = load_background_train(self.n_frames_per_half_match, self.interesting_events, self.frames_per_clip) #always the same for each epoch

            self.sampled_events = sample_training(self.augmented_events, how_many=self.class_samples_per_epoch) #sample with different random offsets for each epoch
            self.sampled_background = sample_background(self.all_background_events, how_many=self.class_samples_per_epoch)
            self.sampled_events = self.sampled_events + self.sampled_background
            random.shuffle(self.sampled_events)
        elif "challenge" in self.split: #challenge split without annotations
            self.sampled_events = load_background_val(self.n_frames_per_half_match, [], self.frames_per_clip, self.test_overlap)
            self.sampled_events.sort(key=sort_function)
        else: #testing split with annotations
            self.interesting_events = load_events(self.matches, self.labels_path, self.frames_per_clip)
            self.sampled_events= load_background_val(self.n_frames_per_half_match, self.interesting_events, self.frames_per_clip, self.test_overlap)
            self.sampled_events.sort(key=sort_function)

        self.preprocessing = transforms.Compose([
            transforms.Resize(self.resize_to), #(360,640) or (224, 398)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        item = self.sampled_events[index]
        url = os.path.join(self.frames_path, item["match"].replace("/","_")+"_"+str(item["half"]))
        frame_indexes = item["frame_indexes"]
        label = item["label"]
        rel_offset = item["rel_offset"]
        frames = []

        for i in range(frame_indexes[0], frame_indexes[1]):
            path = os.path.join(url, str(i).zfill(4) + '.jpg')
            if os.path.isfile(path):
                frames.append(self.preprocessing(Image.open(path)))
        try:
            frames = torch.stack(frames)
        except:
            print(url + " " + str(frame_indexes))

        if len(frames) < (self.frames_per_clip):  # if the center frame was at the beginning (or at the end) of the video, we loaded less frames than necessary
            try:
                frames = torch.cat((frames, frames[-1, :, :, :].unsqueeze(0).repeat(self.frames_per_clip - len(frames), 1, 1, 1)), dim=0)
            except:
                frames = torch.zeros((self.frames_per_clip, 3, self.resize_to[0], self.resize_to[1]))

        return frames, LABELS[label], rel_offset, item["match"], item["half"], frame_indexes[0]

    def __len__(self):
        return len(self.sampled_events)

    def update_background_samples(self):
        self.sampled_events = sample_training(self.augmented_events, how_many=self.class_samples_per_epoch)  # sample with different random offsets for each epoch
        self.sampled_background = sample_background(self.all_background_events, how_many=self.class_samples_per_epoch)
        self.sampled_events = self.sampled_events + self.sampled_background
        random.shuffle(self.sampled_events)
