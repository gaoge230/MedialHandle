import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
from PIL import Image
from torchvision import  transforms

def create_collate_fn(word2idx):
    def collate_fn(dataset):
        ground_truth = {}
        tmp = []
        for fn, img, caption, caplen in dataset:  # caps是一个图像对应的多个文本 格式是[['a',','],['c'],...]
            
            tensor_caption=caption.unsqueeze(0)
            caption = caption.numpy().tolist()
            caption = [w for w in caption if w != word2idx['<pad>']]
            ground_truth[fn] = [caption[:]]
            for cap in [caption]:
                tmp.append([fn, img.unsqueeze(0),tensor_caption, cap, caplen])
        
        dataset = tmp  # dataset此时是一个list [[fn, cap[0], fc_feat, att_feat],[fn, caps[1], fc_feat, att_feat],[fn, caps[2], fc_feat, att_feat]]

        dataset.sort(key=lambda p: len(p[3]),
                     reverse=True)  # 上面的dataset按dataset第二个元素的长度为索引，从大到小排列，这里是按文本长度从大到小排列   
        fns, imgs, tensor_captions,caps, caplens = zip(*dataset)
        imgs = torch.cat((imgs), dim=0)

        lengths = [min(len(c), 52) for c in caps]
        caps_tensor = torch.cat((tensor_captions),dim=0)  # (batch,52)
        for i, c in enumerate(caps):
            end_cap = lengths[i]
            caps_tensor[i, :end_cap] = torch.LongTensor(c[:end_cap])

        lengths = torch.LongTensor(lengths)
        lengths = lengths.unsqueeze(1)
        return fns, imgs, caps_tensor, lengths, ground_truth  # ground_truth的格式{'filename':[['a',','],['c'],...]}
    return collate_fn


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_name, split, loadpath,
                 transform=None):
        """
        :param data_folder: 存放数据的文件夹
        :param data_name: 已处理数据集的基名称
        :param split: split, one of 'TRAIN', 'VAL'or 'TEST'
        :param transform:图像转换
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        
        self.transform_RandomRotation = transforms.Compose([transforms.RandomRotation(360)])

        self.transform_RandomHorizontalFlip = transforms.Compose([transforms.RandomHorizontalFlip(0.5)])

        self.transform_normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])
        if self.split in {'VAL', 'TEST'}:

            image = Image.open(loadpath)
            img = image.resize((256, 256), Image.ANTIALIAS)
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            all_feats = torch.Tensor(img).unsqueeze(0)
            
            self.all_feats=all_feats
            # 一个图像对应几个字幕
            self.cpi = 1
            # Load encoded captions (completely into memory)
            file_name_val = []



    #   加对比损失 
    def __getitem__(self, i):

        all_feats =torch.FloatTensor(self.all_feats[i]/ 255.)
        return all_feats

#     def __getitem__(self, i):
#         # 请记住，第N个标题对应于第个图像（N//captions_per_image）  从第0个开始的
#         img = torch.FloatTensor(self.f[i // self.cpi] / 255.)
        
#         tag= torch.FloatTensor(self.tag[i])
#         all_feats =torch.FloatTensor(self.all_feats[i]/ 255.)

# #         if self.transform is not None:
# #             img = self.transform(img)
#         fn = self.fns[i]

#         caption = torch.LongTensor(self.captions[i])
#         caplen = torch.LongTensor([self.caplens[i]])
#         if self.split is 'TRAIN':
#             return tag,all_feats,fn, img, caption, caplen
#         else:
#             # 取出测试集的图像名字，方便使用指标评估
#             with open('/zengxh_fix/hhy/code/image_caption_tutorial/image_caption_second_point/features/val.txt', 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
#                 line = lines[i].strip().split('\t')
#                 cocoid = torch.zeros(1) + (int(line[0]))

#             all_captions = torch.LongTensor(
#                 self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])  # list切片索引左闭右开
#             return tag,all_feats,fn, img, caption, caplen, all_captions, cocoid

    def __len__(self):
        return 1


def get_dataloader(data_name, split,workers, batch_size,word2idx,folder):
    dataset = CaptionDataset(data_name, split,data_folder=folder)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True,\
            collate_fn = create_collate_fn(word2idx))
    return dataloader