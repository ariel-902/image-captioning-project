from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv
import torch

from PIL import Image
import numpy as np
import random
import os

# from transformers import BertTokenizer

from utils import nested_tensor_from_tensor_list, read_json
# from config import encoder_Config
from tokenizer import *

MAX_DIM = 224


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Resize((224,224)), 
    tv.transforms.Lambda(under_max),

    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.Resize((224,224)), 
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class Caption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()
        # print(root)
        self.root = root
        self.transform = transform
        self.annot = [(self._process(val['image_id'], ann['images']), val['caption'])
                      for val in ann['annotations']]
        # print(len(ann['annotations']))

        # self.max_length = 0
        # for val in ann['annotations']:
        #     if len(val['caption']) > self.max_length:
        #         self.max_length = len(val['caption'])
        # self.max_length = self.max_length + 1

        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)
        self.tokenizer = BPETokenizer(encoder_file='./encoder.json', vocab_file='./vocab.bpe')

        self.max_length = max_length + 1
    

    def _process(self, image_id, images):
        for val in images:
            if val['id'] == image_id:
                return val['file_name']

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id)).convert("RGB")
        # print(type(image))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode(caption)
        # print(caption_encoded)
        cross_caption_encoded = [50256] + caption_encoded + [50256]
        model_caption_encoded = [50256] + caption_encoded + [50256]

        # padding
        cross_caption_encoded += [-100] * (self.max_length - len(cross_caption_encoded))
        model_caption_encoded += [50256] * (self.max_length - len(model_caption_encoded))

        cross_caption_encoded = torch.tensor(cross_caption_encoded, dtype=torch.long)
        model_caption_encoded = torch.tensor(model_caption_encoded, dtype=torch.long)
        # caption_encoded = caption_encoded.ToTensor()
        # print(caption_encoded)
    

        return image.tensors.squeeze(0), image.mask.squeeze(0), cross_caption_encoded, model_caption_encoded, image_id


def build_dataset(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'images', 'train')
        train_file = os.path.join(config.dir, 'train.json')
        data = Caption(train_dir, read_json(train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training')
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.dir, 'images', 'val')
        val_file = os.path.join(config.dir, 'val.json')
        data = Caption(val_dir, read_json(val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")