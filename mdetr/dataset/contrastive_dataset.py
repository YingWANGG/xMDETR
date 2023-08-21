import json
import random
import sys
import re
import traceback
from base64 import b64decode

from random import randint, shuffle
from random import random as rand

from .coco import make_coco_transforms
from util.misc import NestedTensor

import torch
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


from dataset.dist_dataset import DistLineReadingDataset


class ImageMultiTextDataset(Dataset):
    def __init__(self, data_path, batch_size=8, tokenizer=None):

        with open(data_path) as f:
            self.data = json.load(f)

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.add_eos = True  # always add eos
        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.PAD_mask = -100  # loss will ignore this
        self.max_words = 64
        self.max_tokens = 64

        self.image_res = 224
        self.patch_size = 32
        assert self.image_res % self.patch_size == 0
        self.num_patch = int(self.image_res / self.patch_size)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ann = self.data[index]
        image = Image.open(ann["image"]).convert('RGB')
        transform = make_coco_transforms("contrastive", cautious=False)
        image = transform(image)
        caption, caption_2 = ann["caption_en"], ann["caption_new"]
    
        text_ids, text_atts = self.preprocess(caption)
        text_ids_2, text_atts_2 = self.preprocess(caption_2)

        return image, text_ids, text_atts, text_ids_2, text_atts_2

    def preprocess(self, text):
        text = self.pre_caption(text, self.max_words)  # be careful, if text is '', it will cause error
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

        if self.add_eos:
            tokens = tokens[:self.max_tokens - 1]
            tokens += [self.eos_token]

        n_tokens = len(tokens)
        assert n_tokens >= 2, "len(word tokens) < 2"

        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int

        # pad
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        return text_ids, text_atts

    def collate_fn(self, batch):
        batch_tensors = []
        tensor_list = list(zip(*batch))
        batch_tensors.append(NestedTensor.from_tensor_list(tensor_list[0], False))
        for x in tensor_list[1:]:
            if x[0] is None:
                batch_tensors.append(None)
            elif isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))
            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))

        return batch_tensors

    def pre_caption(self, caption, max_words):
        caption_raw = caption
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            ' ',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        if not len(caption):
            raise ValueError(f"pre_caption yields invalid text (raw: {caption_raw})")

        return caption
    
        