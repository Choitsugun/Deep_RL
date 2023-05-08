from torch.utils.data import DataLoader, Dataset
from autoencoder import AutoEncoder
import torch.nn.functional as F
import codecs
import torch

def load_sali(sali_path, logger):
    logger.info("Loading the dull resp list")
    l_sali = [line.strip() for line in codecs.open(sali_path, 'r', 'utf-8').readlines() if line.strip()]
    logger.info("Construct the MyDataset of dull resp list")
    dataset = MyDataset(l_sali)

    return dataset


def tokenizer_process(batch, tokenizer, device):
    b_sali = batch
    b_sali = tokenizer(b_sali, padding=True, return_tensors="pt").to(device)

    return b_sali


class MyDataset(Dataset):
    def __init__(self, sali):
        self.sali = sali

    def __getitem__(self, index):
        b_sali = self.sali[index]
        return b_sali

    def __len__(self):
        return len(self.sali)


class Collater():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        sali = tokenizer_process(batch, self.tokenizer, self.device)
        return sali


class CosDis_reward():
    def __init__(self, tokenizer, args, logger):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = args.device
        self.AtEc = AutoEncoder(args.dime_model, len(tokenizer), tokenizer.pad_token_id, args.device)
        self.AtEc.load_state_dict(torch.load(args.AtEc_checkp)["AtEc"])
        self.AtEc.to(args.device)
        self.AtEc.eval()
        logger.info("Restored the AutoEncoder model from the check point")

        sali = load_sali(args.sali_path, logger)
        collate_fn = Collater(tokenizer, args.device)
        sali_dataloader = DataLoader(sali, batch_size=args.batch_size, collate_fn=collate_fn)
        self.hiddens = torch.tensor([]).to(args.device)
        logger.info("Starting compute the vector spaces of the dull resp list")

        for batch_idx, batch_data in enumerate(sali_dataloader):
            seq = batch_data
            output = self.AtEc.forward(seq, if_train=False)
            self.hiddens = torch.cat((self.hiddens, output), dim=0)    # N C

    def get_cosdis_reward(self, utte):
        utte_enco = self.tokenizer(utte, return_tensors="pt").to(self.device)
        utte_vec = self.AtEc.forward(utte_enco, if_train=False)    # 1 C
        u_s = F.normalize(utte_vec, dim=-1)
        h_s = F.normalize(self.hiddens, dim=-1)
        cos_dis = torch.abs(h_s.matmul(u_s.permute([1, 0])))    # N 1
        cos_dis = torch.max(cos_dis).item()

        return 1-cos_dis