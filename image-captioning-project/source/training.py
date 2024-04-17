import collections
import math
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision as tv
import torch.nn.functional as F

from PIL import Image
import numpy as np
import random
import os
from p2_data import *
from transformer import *
from tqdm.auto import tqdm
from decoder import *
from tokenizer import BPETokenizer
import loralib as lora
from testing import Dataset
# from torch.cuda.amp import autocast
from p2_evaluate import *
# import gc
# from torchsummary import summary

def sameseed(myseed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
sameseed(6666)

# encoder_config = encoder_Config()
cfg = Config()
tokenizer = BPETokenizer(encoder_file='./encoder.json', vocab_file='./vocab.bpe')

# load data
train_set = build_dataset(cfg, mode='training')
train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=False)
IMAGE_PATH = './hw3_data/p2_data/images/val/'
valid_set = Dataset(root=IMAGE_PATH)
inference_step = 60

# training hyperparameter
batch_size = cfg.batch_size
device = cfg.device
epoch_num = 20
lr_rate = cfg.lr
weight_decay = cfg.weight_decay

# model
model = Transformer(cfg).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
# PATH = './large_testing_model.pth'
# model = transformer.Transformer(cfg).to(device)
# model.load_state_dict(torch.load(PATH), strict=False)

# load model
# PATH = './p3_large_lora_10.ckpt'
# model.load_state_dict(torch.load(PATH), strict=False)
# if cfg.PEFT_mode == 'lora':
#     model.load_state_dict(torch.load('p3_lora_parameter_10.ckpt'), strict=False)

# validation used material
annotations = readJSON('./hw3_data/p2_data/val.json')
gts = getGTCaptions(annotations)
# image_root = './hw3_data/p2_data/images/val/'


# freeze 
if cfg.PEFT_mode == 'lora':
    lora.mark_only_lora_as_trainable(model)
    for name, param in model.named_parameters():
        if 'crossattn' in name:
            param.requires_grad = True
        if 'encoder.linear' in name:
            param.requires_grad = True

if cfg.PEFT_mode == 'adapter':
    for name, param in model.named_parameters():
        # print(name)
        param.requires_grad = False
        if 'crossattn' in name:
            param.requires_grad = True
        if 'adapter' in name:
            param.requires_grad = True
        if 'adapter_mlp' in name:
            param.requires_grad = True
        if 'encoder.linear' in name:
            param.requires_grad = True

if cfg.PEFT_mode == 'prefix':
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'crossattn' in name:
            param.requires_grad = True
        if 'pre' in name:
            param.requires_grad = True
        if 'encoder.linear' in name:
            param.requires_grad = True



trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
# print(trainable_weights)

print("Total params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("PEFT mode: ", cfg.PEFT_mode)

# min_loss = 10000
max_clip = 0.0
max_cider = 0.0
# training
for epoch in range(epoch_num):
    epoch_loss = 0.0
    total = len(train_loader)
    model.train()
    for batch in tqdm(train_loader):

        img, img_mask, cross_caption_encoded, model_caption_encoded,_= batch # img = [32, 3, 224, 224]
        img = img.to(device)
        cross_caption_encoded = cross_caption_encoded.to(device)
        model_caption_encoded = model_caption_encoded.to(device)

        # with autocast(enabled=False):

            # logits = model(img, model_caption_encoded[:, :-1])[:,:cfg.max_position_embeddings,:] 

        logits = model(img, model_caption_encoded[:, :-1])

        loss = criterion(logits.permute(0,2,1), cross_caption_encoded[:,1:])
        loss_value = loss.item()
        epoch_loss += loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"training loss in epoch {epoch+1}:", epoch_loss / total)

    # validate
    model.eval()
    result_dict = {}
    for img, image_id in tqdm(valid_set):
        # count += 1
        caption = torch.zeros((1, inference_step + 1), dtype=torch.long)
        caption[:, 0] = 50256
        for i in range(inference_step - 1):
            img = img.to(device)
            caption = caption.to(device)
            with torch.no_grad():
                predictions = model(img, caption)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 50256:
                caption[:, i+1] = predicted_id[0]
                result = caption[:,1:i+1]
                result = tokenizer.decode(result[0].tolist())
                result_dict[image_id.split(".")[0]] = result
                # print(result_dict)
                # print(f"image:{image_id}:", result)
                break

            caption[:, i+1] = predicted_id[0]
            if (i == inference_step - 2):
                result = caption[:,1:i+1]
                result = tokenizer.decode(result[0].tolist())
                result_dict[image_id.split(".")[0]] = result
        
    # CIDErScore
    cider_score = CIDERScore()(result_dict, gts)
    # CLIPScore
    clip_score = CLIPScore()(result_dict, IMAGE_PATH)
    
    print(f"validation from epoch {epoch+1}: CIDEr: {cider_score} | CLIPScore: {clip_score}")


    trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
    save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights}
    # torch.save(save_weights, f"large_testing_model.pth")
    # torch.save(save_weights, f"deit_{cfg.PEFT_mode}_{epoch+1}.pth")
    # torch.save(save_weights, f"base_{cfg.PEFT_mode}_{epoch + 1}.ckpt")

    # if cfg.PEFT_mode == 'lora':
        # torch.save(lora.lora_state_dict(model), f"p3_large_{cfg.PEFT_mode}_parameter_{epoch+1}.ckpt"

    # if ((cider_score > max_cider)):
    #     max_cider = cider_score
    #     torch.save(save_weights, f"base_{cfg.PEFT_mode}_best.ckpt")
        # if cfg.PEFT_mode == 'lora':
            # torch.save(lora.lora_state_dict(model), f"p3_large_{cfg.PEFT_mode}_parameter_best.ckpt")

    # gc.collect()
    # torch.cuda.empty_cache()




