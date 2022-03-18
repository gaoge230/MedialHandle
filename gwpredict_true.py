import numpy as np
def getYZresult(path):
    import numpy as np
    import pandas as pd
    import json
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from tqdm import tqdm
    from sklearn.neighbors.kde import KernelDensity
    from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
    from sklearn import preprocessing
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import torch.nn as nn

    def get_loader(data_path, batch_size, mode='train'):
        """Build and return data loader."""

        dataset = HeartLoader(data_path, mode)

        shuffle = False
        if mode == 'train':
            shuffle = True

        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle)
        return data_loader

    class HeartLoader():
        def __init__(self, data_path, mode="train"):
            self.mode = mode
            data = pd.read_csv(data_path, header=None)
            data = np.array(data)
            # data = np.load(data_path)
            # print(data)
            # print(data.shape)

            labels = data[:, 0]
            features = data[:, 1:]
            # print("labels", labels.shape)
            # print("features",features)


            self.train = features[0:1134]   # 训练集
            self.test = features[1134:]     # 测试集
            self.train_label = labels[0:1134]  # 训练集标签 (800,1)
            self.test_label = labels[1134:]    # 测试集标签 (200,1)

        def __getitem__(self, index):
            if self.mode == "train":
                return np.float32(self.train[index]), np.float32(self.train_label[index])
            else:
                return np.float32(self.test[index]), np.float32(self.test_label[index])

        def __len__(self):
            if self.mode == "train":
                return self.train.shape[0]
            else:
                return self.test.shape[0]


    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()

            self.enc_1 = nn.Linear(14, 7)
            self.enc_2 = nn.Linear(7, 3)
            self.enc_3 = nn.Linear(3, 1)

            self.act = nn.Tanh()
            self.act_s = nn.Sigmoid()

            self.dec_3 = nn.Linear(1, 3)
            self.dec_2 = nn.Linear(3, 7)
            self.dec_1 = nn.Linear(7, 14)

            # self.seq = nn.Sequential(
            #               nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5),# 改out_channels:2的倍数(卷积核个数)，多层设置
            #               nn.Conv1d(in_channels=8, out_channels=5, kernel_size=5),
            #               nn.Conv1d(in_channels=5, out_channels=1, kernel_size=5),
            #            )
            self.seq = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=12, kernel_size=5),  # 改out_channels:2的倍数(卷积核个数)，多层设置
                nn.Conv1d(in_channels=12, out_channels=10, kernel_size=5),
                nn.Conv1d(in_channels=10, out_channels=8, kernel_size=3),
                nn.Conv1d(in_channels=8, out_channels=6, kernel_size=3),
                nn.Conv1d(in_channels=6, out_channels=4, kernel_size=1),
                nn.Conv1d(in_channels=4, out_channels=2, kernel_size=1),
                nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1),
            )


        def forward(self, x):
            conv1d = self.seq(x.unsqueeze(dim=1))
            enc_1 = self.enc_1(x)
            # enc_1 = self.act(enc_1)
            enc_2 = self.enc_2(enc_1)
            # enc = self.act(enc)
            enc_3 = self.enc_3(enc_2)

            dec_3 = self.dec_3(enc_3)
            # dec = self.act(z_1)
            dec_2 = self.dec_2(dec_3)
            # dec = self.act_s(dec)
            dec_1 = self.dec_1(dec_2)

            return enc_1, enc_2, enc_3, dec_3, dec_2, dec_1, conv1d.squeeze(1)

    def relative_euclidean_distance(a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    # 均方误差
    loss_fn = nn.MSELoss(reduction='mean')   # 获取均方误差的平均值
    def loss_function(a, b, input_data, dec_1, enc_1, dec_2, enc_2, dec_3, conv1d):
        # b是label
        loss1 = loss_fn(a, b)
        loss2 = loss_fn(input_data, dec_1)
        loss3 = loss_fn(enc_1, dec_2)
        loss4 = loss_fn(enc_2, dec_3)
        lossconv1d = loss_fn(conv1d, b.unsqueeze(dim=1))
        # print(lossconv1d)
        return loss1 + 0.01*loss2 + 0.01*loss3 + 0.01*loss4 + 0.01*lossconv1d

    data_path = path
    batch_size = 20
    learn_rate = 0.001
    iter_per_epoch = 10
    # 使用模型
    vae = VAE()
    loss_list = []

    data_loader_train = get_loader(data_path, batch_size, mode='train')
    optimizer = torch.optim.Adam(vae.parameters(), lr=learn_rate)
    for epoch in range(iter_per_epoch):
        temploss = []
        # print(temploss)
        for i, (input_data, labels) in enumerate(data_loader_train):
            # print(labels.shape)
            enc_1, enc_2, enc_3, dec_3, dec_2, dec_1, conv1d = vae(input_data)
            # print(conv1d.shape)
            optimizer.zero_grad()
            final = enc_3
            # print(labels[0])
            # print(final.size())
            loss = loss_function(final, labels, input_data, dec_1, enc_1, dec_2, enc_2, dec_3, conv1d)
            temploss.append(loss.item())
            # print(epoch, loss)
            loss.backward()
            optimizer.step()
            # print(loss)


        # print(temploss)
        print(epoch+1, np.mean(temploss))
        loss_list.append(np.mean(temploss))



    # final_loss = [w.item() for w in loss_list]
    # with open("result/epoch_loss.json", 'w', encoding='utf-8') as j:
    #     json.dump(final_loss, j, ensure_ascii=False)

    batch_size_test = 3000

    enc_3 = 0
    data_loader_test = get_loader(data_path, 3000,  mode='test')
    for i, (input_data, labels) in enumerate(data_loader_test):
        enc_1, enc_2, enc_3, dec_3, dec_2, dec_1, conv1d = vae(input_data)
        # print(enc_3)
        # loss = loss_function(enc_3[0], labels[0])
        res = mean_squared_error(labels.detach().numpy(), enc_3.detach().numpy())
        res1 = mean_absolute_error(labels.detach().numpy(), enc_3.detach().numpy())
        res2 = r2_score(labels.detach().numpy(), enc_3.detach().numpy())
        # print(loss)
        print(res)
        print(res1)
        print(res2)

    return enc_3.detach().numpy().ravel().astype(np.int)
    # return enc_3

if __name__ == "__main__":
    res = getYZresult(r"C:\Users\chenxinxin\Desktop\回校文件\AnoDetection\model\data\allnormalsort_yunzhou_forModel.csv")
    print(res.detach().numpy().ravel().astype(np.int))