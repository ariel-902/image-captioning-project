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
# from p2_evaluate import *
import sys
# from decoder import *


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
                        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

        # print(f'Number of images is {len(self.images)}')

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


# training hyperparameter
def main(IMAGE_PATH, OUTPUT_PATH, DECODER_WEIGHT_PATH):

    cfg = Config(checkpoint=DECODER_WEIGHT_PATH)  
    batch_size = 1
    device = cfg.device

    # load data
    # IMAGE_PATH = './hw3_data/p2_data/images/val/'
    valid_set = Dataset(root=IMAGE_PATH)

    # validation used material
    # annotations = readJSON('./hw3_data/p2_data/val.json')
    # gts = getGTCaptions(annotations)

    # import model # learning_rate = 3e-5
    model = Transformer(cfg).to(device)
    # MODEL_PATH = './large_lora_9.ckpt'
    # MODEL_PATH = './in12k_lora_7.ckpt'
    MODEL_PATH = './p2_model.ckpt'
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)

    state_dict = torch.load(MODEL_PATH)
    print(sum([p.numel() for n, p in state_dict.items()]))

    tokenizer = cfg.tokenizer

    # inference
    model.eval()
    MAX = 64
    result_dict = {}

    with torch.no_grad():
        for img, image_id in tqdm(valid_set):
            # print(image_id)
            k = 5
            k_prev_words = torch.full((k, 1), 50256, dtype=torch.long) # (k, 1)
            # print(k_prev_words)
            #此时输出序列中只有sostoken
            seqs = k_prev_words #(k, 1)
            seqs = seqs.to(device)
            #初始化scores向量为
            top_k_scores = torch.zeros(k, 1)
            complete_seqs = list()
            complete_seqs_scores = list()

            # step = 1
            # hidden = torch.zeros(1, k, hidden_size) # h_0: (1, k, hidden_size)
            for i in range(MAX - 1):
                img = img.to(device)
                # print(k_prev_words)
                # print("img:", img)
                k_prev_words = k_prev_words.to(device)
                predictions = model(img, k_prev_words) # k * max_positioning * 50256
                # outputs, hidden = decoder(k_prev_words) # outputs: (k, seq_len, vocab_size)
                next_token_logits = predictions[:,-1,:] #(k, vocab_size)
                next_token_logits = F.softmax(next_token_logits, dim=-1)
                # print(next_token_logits)

                if (i == 0):
                    #因为最开始解码的时候只有一个结点sos,所以只需要取其中一个结点计算topk
                    top_k_scores, top_k_words = next_token_logits[0].topk(k, dim=0, largest=True, sorted=True)
                else:
                    #此时要先展开再计算topk,如上图所示。
                    probs = top_k_scores.unsqueeze(1) * next_token_logits 
                    top_k_scores, top_k_words = probs.view(-1).topk(k, dim=0, largest=True, sorted=True)

                # prev_word_inds = top_k_words // cfg.vocab_size #(k)实际是beam_id
                prev_word_inds = torch.div(top_k_words, cfg.vocab_size, rounding_mode='trunc')
                next_word_inds = top_k_words % cfg.vocab_size  #(k)实是token_id
                # seqs: (k, step) ==> (k, step+1)

                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                # print(seqs)

                #当前输出的单词不是eos的有哪些(输出其在next_wod_inds中的位置,实际是beam_id)
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != 50256]

                #输出已经遇到eos的句子的beamid(即seqs中的句子索
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist()) #加入句子
                    complete_seqs_scores.extend(top_k_scores[complete_inds]) #加入句子对应的累加og_prob
                
                #减掉已经完成的句子的数量更新k,次就不用执行那么多op了,因为若干句子已经被解码出来
                k -= len(complete_inds)

                if k == 0:#完成
                    break

                #更新一次代数据,仅专注于那些还没完成的句子
                seqs = seqs[incomplete_inds]
                # hidden = hidden[prev_word_inds [incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds] #(s, 1) s <k
                # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1) #(s, 1) s<k
                k_prev_words = seqs
                if (i == (MAX - 2)):
                    complete_seqs.extend(seqs.tolist())
                    complete_seqs_scores.extend(top_k_scores)
                    break

            # print(complete_seqs_scores)
            max_index = complete_seqs_scores.index(max(complete_seqs_scores))#寻找score最大的序列
            #有些许问题,在训练初期一直碰不到eos时,此时complete_seqs为空
            seq = complete_seqs[max_index]

            # print(seq)
            caption = seq[1:-1]
            # print(caption)
            # print(tokenizer.decode(caption))
            result = tokenizer.decode(caption)
            result_dict[image_id.split(".")[0]] = result


    # CIDErScore
    # cider_score = CIDERScore()(result_dict, gts)
    # CLIPScore
    # clip_score = CLIPScore()(result_dict, IMAGE_PATH)
    
    # print(f"validation: CIDEr: {cider_score} | CLIPScore: {clip_score}")

    with open(OUTPUT_PATH, 'w') as fp:
        json.dump(result_dict, fp)


if __name__ == '__main__':

    IMAGE_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]
    DECODER_WEIGHT = sys.argv[3]

    # IMAGE_PATH = './hw3_data/p2_data/images/val/'
    # OUTPUT_PATH = 'inki_result.json'
    # DECODER_WEIGHT = './hw3_data/p2_data/decoder_model.bin'

    main(IMAGE_PATH, OUTPUT_PATH, DECODER_WEIGHT)






