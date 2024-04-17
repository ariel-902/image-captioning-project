from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision as tv
import torch.nn.functional as F

from PIL import Image
import numpy as np
import random
import os
from p2_data import *
# from config import encoder_Config
from transformer import *
from tqdm.auto import tqdm
# from p2_evaluate import *
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize

cfg = Config()

class Dataset():
    def __init__(self, root):
        # IMAGE_PATH = root
        self.root = root
        self.images = sorted(os.listdir(self.root))
        # images = glob.glob(os.path.join(IMAGE_PATH, '*.jpg'))
        # self.images = root
        self.transform = tv.transforms.Compose([
                        tv.transforms.Resize((224,224)),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
                             0.5, 0.5, 0.5])
                    ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(os.path.join(self.root,img_path))
        if image.mode != 'RGB':
            image = image.convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)


        return image, img_path

def sameseed(myseed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

sameseed(6666)


def register_attention_hook(model, features, feature_size):
    def hook_decoder(module, ins, outs):
        # print(ins)
        att = module.att # att = [1, 12, 61, 257]
        # print(outs.size()) # batch * 61 * 768
        # print(att.view(1,61,4).size())
        att = torch.sum(att, dim=1)
        # print(att.size())
        features.append(att.detach().cpu())
        # features.append(outs.detach().cpu())
    # handle_decoder = model.decoder.h.Block.CrossAttention.register_forward_hook(
        # hook_decoder)
    handle_decoder = model.decoder.block_layer[-1].crossattn.register_forward_hook(hook_decoder)

    return [handle_decoder]

def vis_atten_map(atten_mat, ids, feature_size, image_fn, image_path):
    # print(atten_mat.shape)
    nrows = len(ids) // 5 if len(ids) % 5 == 0 else len(ids) // 5 + 1
    ncols = 5
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 8))
    feature_size = (16, 16)  # H/patch size, W/patch size = feature size
    for i, id in enumerate(ids):
        attn_vector = atten_mat[:, i - 1, 1:]
        attn_map = torch.reshape(attn_vector, feature_size)
        attn_map -= torch.min(attn_map)
        attn_map /= torch.max(attn_map)
        # print(torch.min(attn_map), torch.max(attn_map))
        # print(attn_map.size())
        im = Image.open(image_path)
        size = im.size
        mask = resize(attn_map.unsqueeze(0), [size[1], size[0]]).squeeze(0)
        mask = np.uint8(mask * 255)
        # print(mask.shape)
        ax[i // 5][i % 5].imshow(im)
        if i == 0:
            ax[i // 5][i % 5].set_title('<|endoftext|>')
        elif i == len(ids) - 1:
            ax[i // 5][i % 5].set_title('<|endoftext|>')
            ax[i // 5][i % 5].imshow(mask, alpha=0.7, cmap='jet')
        else:
            ax[i // 5][i % 5].set_title(id)
            ax[i // 5][i % 5].imshow(mask, alpha=0.7, cmap='jet')
        ax[i // 5][i % 5].axis('off')
    for i in range(len(ids), nrows * ncols):
        ax[i // 5][i % 5].axis('off')
    # plt.savefig(args.output_dir / image_fn)
    plt.savefig(image_fn)

def greedy(model, img, inference_step):

    device = 'cuda'
    caption = torch.zeros((1, inference_step + 1), dtype=torch.long)
    caption[:, 0] = 50256
    for i in range(inference_step - 1):
        img = img.to(device)
        caption = caption.to(device) 
        predictions = model(img, caption)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 50256:
            caption[:, i+1] = predicted_id[0]
            result = caption[:,1:i+1]
            # result = tokenizer.decode(result[0].tolist())
            # result_dict[image_id.split(".")[0]] = result
            # print(result_dict)
            # print(f"image:{image_id}:", result)
            break

        caption[:, i+1] = predicted_id[0]
        if (i == inference_step - 2):
            result = caption[:,1:i+1]
            # result = tokenizer.decode(result[0].tolist())
            # result_dict[image_id.split(".")[0]] = result
        
    return result

def main():

    device = cfg.device
    batch_size = 1
    # load data
    IMAGE_PATH = './p3_image/'
    valid_set = Dataset(root=IMAGE_PATH)

    # load model
    model = Transformer(cfg).to(device)
    PATH = './large_lora_9.ckpt'
    model.load_state_dict(torch.load(PATH), strict=False)

    tokenizer = cfg.tokenizer
    model.eval()

    inference_step = 60

    for img, image_path in tqdm(valid_set):
        # print(image_path)
        image_id = image_path.split("/")[-1]
        features, feature_size = [], []
        to_rm_l = register_attention_hook(model, features, feature_size)
        output_ids = greedy(model, img, inference_step)

        result = tokenizer.decode(output_ids[0].tolist())
        output_tokens = ['<|endoftext|>'] + result.split(" ") + ['<|endoftext|>']
        # print(output_tokens)

        # visualize
        attention_matrix = features[-1]
        # print(attention_matrix.size()) # 1 * 61 * 768
        # vis_atten_map(attention_matrix, output_tokens, feature_sizes,
                    #   image_id, (args.image_dir / name).with_suffix('.jpg'))

        output_path = './p3_draw_result_softmax/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path = os.path.join(output_path, image_id)
        image_path = os.path.join(IMAGE_PATH, image_path)
        vis_atten_map(attention_matrix, output_tokens, feature_size, 
                        output_path,image_path)

        for handle in to_rm_l:
            handle.remove()

if __name__ == '__main__':
    main()
