from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy

class Deep_Q_net(nn.Module):
    def __init__(self, args, logger, if_train=True, graine_sele="fgr"):
        super().__init__()

        if if_train:
            self.mse = nn.MSELoss()
            self.gama = args.gama
            self.qnet = BertModel.from_pretrained(args.DQN_pretrained)
            self.qtar = copy.deepcopy(self.qnet)
            d_model = self.qnet.config.hidden_size
            self.ff1 = nn.Linear(in_features=d_model, out_features=int(d_model/2))
            self.ff2 = nn.Linear(in_features=int(d_model/2), out_features=1)
            logger.info("Initialized the DQN from the pretrained weight")
        else:
            self.device = args.device
            self.tokenizer = BertTokenizer.from_pretrained(args.DQN_pretrained)

            if graine_sele == "cgr":
                self.n_tile = args.ncla_topic
                self.qnet = BertModel.from_pretrained(args.DQN_cgr_checkp)
                d_model = self.qnet.config.hidden_size
                self.ff1 = nn.Linear(in_features=d_model, out_features=int(d_model/2))
                self.ff1.load_state_dict(torch.load(args.DQN_cgr_checkp + "/ff1.pth")["ff1"])
                self.ff2 = nn.Linear(in_features=int(d_model/2), out_features=1)
                self.ff2.load_state_dict(torch.load(args.DQN_cgr_checkp + "/ff2.pth")["ff2"])
                logger.info("Restored the coarse-grained DQN from the check point")

            if graine_sele == "fgr":
                self.n_tile = args.n_retur_seq
                self.qnet = BertModel.from_pretrained(args.DQN_fgr_checkp)
                d_model = self.qnet.config.hidden_size
                self.ff1 = nn.Linear(in_features=d_model, out_features=int(d_model/2))
                self.ff1.load_state_dict(torch.load(args.DQN_fgr_checkp + "/ff1.pth")["ff1"])
                self.ff2 = nn.Linear(in_features=int(d_model/2), out_features=1)
                self.ff2.load_state_dict(torch.load(args.DQN_fgr_checkp + "/ff2.pth")["ff2"])
                logger.info("Restored the fine-grained DQN from the check point")

            self.qnet.eval()
            self.ff1.eval()
            self.ff2.eval()

    def sync_qnet(self):
        self.qtar = copy.deepcopy(self.qnet)

    def forward(self, sou=None, tar=None, rew=None, S_t=None, A_t=None, if_train=True):
        if if_train:
            score_sou = self.ff2(F.leaky_relu(self.ff1(self.qnet(**sou).last_hidden_state[:,0,:]))).squeeze(-1)  # N
            score_tar = self.ff2(F.leaky_relu(self.ff1(self.qtar(**tar).last_hidden_state[:,0,:]))).squeeze(-1) * self.gama + rew  # N
            score_tar = score_tar.detach()

            return self.mse(score_sou, score_tar)
        else:
            S_t = [" ".join(S_t) for i in range(self.n_tile)]
            sou = self.tokenizer(S_t, A_t, padding=True, return_tensors="pt").to(self.device)

            return self.ff2(F.leaky_relu(self.ff1(self.qnet(**sou).last_hidden_state[:,0,:]))).squeeze(-1)