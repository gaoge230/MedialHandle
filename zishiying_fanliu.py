def getADresult(path):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from sklearn.neighbors.kde import KernelDensity
    from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import torch.nn as nn

    def get_loader(data_path, batch_size, N_train, mode='train'):
        """Build and return data loader."""

        dataset = HeartLoader(data_path, N_train, mode)

        shuffle = False
        if mode == 'train':
            shuffle = True

        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle)
        return data_loader

    class HeartLoader(object):
        def __init__(self, data_path, N_train, mode="train"):
            self.mode = mode
            data = pd.read_csv(data_path,header=None)
            data = np.array(data)
            # data = np.load(data_path)
            # print(data)
            # print(data.shape)            # 1000, 16

            labels = data[:, -1]
            features = data[:, :-1]
            N, D = features.shape
            # print(features)


            normal_data = features[labels == 1]  # 选择正常的数据
            # print(normal_data.shape)
            normal_labels = labels[labels == 1]

            # N_normal = normal_data.shape[0]

            attack_data = features[labels == 0]  # 选择异常的数据
            # print(attack_data.shape)
            attack_labels = labels[labels == 0]

            N_attack = attack_data.shape[0]

            randIdx = np.arange(N_attack)  # 将数据的顺序打乱
            np.random.shuffle(randIdx)

            # print(N_attack, N_train)

            # 训练集只有正常标签，测试集正常数据+异常数据
            self.N_train = N_train
            self.train = attack_data[randIdx[:self.N_train]]
            self.train_labels = attack_labels[randIdx[:self.N_train]]

            self.test = attack_data[randIdx[self.N_train:]]
            self.test_labels = attack_labels[randIdx[self.N_train:]]

            self.test = np.concatenate((self.test, normal_data), axis=0)
            self.test_labels = np.concatenate((self.test_labels, normal_labels), axis=0)

            # print(self.train)

        def __len__(self):
            """
            Number of images in the object dataset.
            """
            if self.mode == "train":
                return self.train.shape[0]
            else:
                return self.test.shape[0]

        def __getitem__(self, index):
            if self.mode == "train":
                return np.float32(self.train[index]), np.float32(self.train_labels[index])
            else:
                return np.float32(self.test[index]), np.float32(self.test_labels[index])

    def loss_function(recon_x, x, mu, logvar, enc, z,  enc_1, z_1):
        # loss = loss_function(dec, input_data, mu, log_var, enc, z, enc_1, z_1)
        criterion_elementwise_mean = nn.MSELoss(reduction='sum')
        BCE_x = criterion_elementwise_mean(recon_x,x)
        BCE_z = criterion_elementwise_mean(enc,z)
        BCE_z_1 = criterion_elementwise_mean(enc_1,z_1)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


        all = BCE_x + BCE_z + BCE_z_1
        return (1 - BCE_x / all) * BCE_x + (1 - BCE_z / all) * BCE_z + (1 - BCE_z_1 / all) * BCE_z_1 + KLD
        # all = BCE_x + BCE_z_1
        # return (1 - BCE_x / all) * BCE_x + (1 - BCE_z_1 / all) * BCE_z_1 + KLD

    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()
            # self.enc_1 = nn.Linear(22, 20)
            # self.enc = nn.Linear(20, 15)
            #
            # self.act = nn.Tanh()
            # self.act_s = nn.Sigmoid()
            # self.mu = nn.Linear(15, 15)
            # self.log_var = nn.Linear(15, 15)
            #
            # self.z = nn.Linear(15, 15)
            # self.z_1 = nn.Linear(15, 20)
            # self.dec = nn.Linear(20, 22)

            self.enc_1 = nn.Linear(15, 13)
            self.enc = nn.Linear(13, 10)

            self.act = nn.Tanh()
            self.act_s = nn.Sigmoid()
            self.mu = nn.Linear(10, 10)
            self.log_var = nn.Linear(10, 10)

            self.z = nn.Linear(10, 10)
            self.z_1 = nn.Linear(10, 13)
            self.dec = nn.Linear(13, 15)

        def reparameterize(self, mu, log_var):
            std = torch.exp(log_var / 2)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            enc_1 = self.enc_1(x)
            enc = self.act(enc_1)
            enc = self.enc(enc)
            enc = self.act(enc)

            mu = self.mu(enc)
            log_var = self.log_var(enc)
            o = self.reparameterize(mu, log_var)
            z = self.z(o)
            z_1 = self.act(z)
            z_1 = self.z_1(z_1)
            dec = self.act(z_1)
            dec = self.dec(dec)
            dec = self.act_s(dec)
            return enc_1, enc, mu, log_var, o, z, z_1, dec

    def relative_euclidean_distance(a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)


    data_path = path

    batch_size = 64
    learn_rate = 0.0001
    All_train = 900

    Ratio = 0.1
    iter_per_epoch = 1
    Average_cycle = 1
    result = []
    diff_quantity_result = []
    for i in range(1):
        N_train = int(All_train * Ratio * (i+8))      # 把8改为i+2
        result = []
        # print(Ratio * (i+8))
        for i in tqdm(range(Average_cycle)):
            vae = VAE()
            optimizer = torch.optim.Adam(vae.parameters(), lr=learn_rate)
            data_loader_train = get_loader(data_path, batch_size, N_train, mode='train')
            for i in range(iter_per_epoch):
                for j, (input_data, labels) in enumerate(data_loader_train):
                    enc_1, enc, mu, log_var, o, z, z_1, dec = vae(input_data)
                    optimizer.zero_grad()
                    loss = loss_function(dec, input_data, mu, log_var, enc, z, enc_1, z_1)
                    loss.backward()
                    optimizer.step()

            batch_size = 1000
            data_loader_train = get_loader(data_path, batch_size, N_train, mode='train')
            train_enc = []
            train_labels = []
            data_loader_test = get_loader(data_path, batch_size, N_train, mode='test')
            test_enc = []
            test_labels = []

            for i, (input_data, labels) in enumerate(data_loader_train):
                enc_1, enc, mu, log_var, o, z, z_1, dec = vae(input_data)
                rec_euclidean = relative_euclidean_distance(input_data, dec)
                # rec_cosine = F.cosine_similarity(input_data, dec, dim=1)

                # enc = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
                enc = torch.cat([enc, rec_euclidean.unsqueeze(-1)], dim=1)
                enc = enc.detach().numpy()

                train_enc.append(enc)
            for i, (input_data, labels) in enumerate(data_loader_test):
                enc_1, enc, mu, log_var, o, z, z_1, dec = vae(input_data)
                rec_euclidean = relative_euclidean_distance(input_data, dec)
                # rec_cosine = F.cosine_similarity(input_data, dec, dim=1)

                # enc = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
                enc = torch.cat([enc, rec_euclidean.unsqueeze(-1)], dim=1)
                enc = enc.detach().numpy()

                test_enc.append(enc)

                test_labels.append(labels.numpy())
            x = train_enc[0]
            kde = KernelDensity(kernel='gaussian', bandwidth=0.000001).fit(x)
            score = kde.score_samples(x)
            k = len(test_enc)
            test_score = []
            for i in range(k):
                score = kde.score_samples(test_enc[i])
                test_score.append(score)
            test_labels = np.concatenate(test_labels, axis=0)
            test_score = np.concatenate(test_score, axis=0)
            s = len(test_labels)
            c = np.sum(test_labels == 1)
            g = c / s

            np.set_printoptions(threshold=np.inf)
            # print(g)
            thresh = np.percentile(test_score, int(g * 100))
            # print(thresh)
            # print(test_score)
            pred = (test_score < thresh).astype(int)
            # print(pred)

            return pred

if __name__ == "__main__":
    res = getADresult(r"C:\Users\chenxinxin\Desktop\回校文件\AnoDetection\model\data\fanliu.csv")
    print(res)