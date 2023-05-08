from cossi_reward_calc import CosDis_reward
from torch.utils.data import Dataset
from topic_pred import Topic
from tqdm import tqdm
import numpy as np
import torch

class Collater():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        sou, tar, rew = tokenizer_process(batch, self.tokenizer, self.device)
        return sou, tar, rew


class MyDataset(Dataset):
    def __init__(self, data_list, rewa_list):
        self.data_list = data_list
        self.rewa_list = rewa_list

    def __getitem__(self, index):
        data = self.data_list[index]
        rewa = self.rewa_list[index]

        return data, rewa

    def __len__(self):
        return len(self.data_list)


class Collater_AtEc():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        utte = tokenizer_process_AtEc(batch, self.tokenizer, self.device)
        return utte


class MyDataset_AtEc(Dataset):
    def __init__(self, utte_pool):
        self.utte_pool = utte_pool

    def __getitem__(self, index):
        utte = self.utte_pool[index]
        return utte

    def __len__(self):
        return len(self.utte_pool)


def appen_utte(dial):
    utte_pool = []

    for uttes in dial:
        for utte in uttes:
            utte_pool.append(utte)

    return utte_pool


def list_add(ls1, ls2):
    ls = []

    for i, uttes in enumerate(tqdm(ls1)):
        ls3 = []

        for ii in range(len(uttes)):
            ls3.append(ls1[i][ii]+ls2[i][ii])

        ls.append(ls3)

    return ls


def core_reward_calcul(dial, cosr):
    dial_cos_reward = []
    dial_res_reward = []

    for uttes in tqdm(dial):
        utte_cos_reward = []
        utte_res_reward = []

        for utte in uttes:
            cosdis_reward = cosr.get_cosdis_reward(utte)
            utte_cos_reward.append(cosdis_reward)
            resp_len = len(utte.split())
            if resp_len < 5:
                utte_res_reward.append(-0.2)
            elif resp_len < 10:
                utte_res_reward.append(0.0)
            elif resp_len < 15:
                utte_res_reward.append(0.2)
            else:
                utte_res_reward.append(0.5)

        # The Cos distance (between utte and dull resp) reward
        dial_cos_reward.append(utte_cos_reward)

        # The resp length reward
        dial_res_reward.append(utte_res_reward)

    return dial_cos_reward, dial_res_reward


def acti_reward_calcul(acti):
    # act_reward_list: a list of classification labels, which question is (2)
    dial_act_reward = []

    for act_l in tqdm(acti):
        new_act_l = [0.0 if i != 2 else 1.0 for i in act_l]
        dial_act_reward.append(new_act_l)

    return dial_act_reward


def emot_reward_calcul(emot):
    # emo_reward_list: a list of classification labels, which no emotion is (0)
    dial_emo_reward = []

    for emo_l in tqdm(emot):
        new_emo_l = [0.0 if i != 6 else 1.0 for i in emo_l]
        dial_emo_reward.append(new_emo_l)

    return dial_emo_reward


def make_reward(dial, acti, emot, cosr, logger):
    logger.info("Starting the calculation of Cos distance reward and response length reward")
    dial_cos_reward, dial_res_reward = core_reward_calcul(dial, cosr)

    logger.info("Starting the calculation of question reward")
    dial_act_reward = acti_reward_calcul(acti)

    logger.info("Starting the calculation of emotion reward")
    dial_emo_reward = emot_reward_calcul(emot)

    logger.info("Starting the summarization of reward")
    rewa = list_add(list_add(list_add(dial_cos_reward, dial_res_reward), dial_act_reward), dial_emo_reward)

    return rewa


def make_format(dial, rewa, num_state, max_len, logger, pred=None):
    logger.info("Starting the data format process")
    data_temp = []
    rewa_temp = []
    data_list = []
    rewa_list = []
    count_less = 0
    count_over = 0

    for i, uttes in enumerate(tqdm(dial)):
        u_len = len(uttes)

        if u_len < 4:
            assert len(rewa[i]) == u_len
            count_less = count_less + 1
        else:
            for ii in range(u_len-3):
                ls = []

                # make S_t
                if ii + 0 < num_state:
                    ls.append(" ".join(uttes[:ii+1]))
                else:
                    ls.append(" ".join(uttes[ii+1-num_state:ii+1]))

                # make A_t
                ls.append(uttes[ii+1])

                # make S_t+1
                if ii + 2 < num_state:
                    ls.append(" ".join(uttes[:ii+3]))
                else:
                    ls.append(" ".join(uttes[ii+3-num_state:ii+3]))

                # make A_t+1
                ls.append(uttes[ii+3])

                # data_temp shape: num_state=3, then [[u1 u2 u3,u4,u3 u4 u5,u6], ...] = N*4
                # rewa_temp shape: [..., ...] = N
                data_temp.append(ls)
                rewa_temp.append(rewa[i][ii+1])

    for i, state_action_pair in enumerate(tqdm(data_temp)):
        if len(" ".join(state_action_pair).split()) > max_len:
            count_over = count_over + 1
        else:
            if pred is not None:
                A_t  = state_action_pair[1]
                A_tt = state_action_pair[3]
                state_action_pair[1] = pred.predict(A_t)
                state_action_pair[3] = pred.predict(A_tt)

            data_list.append(state_action_pair)
            rewa_list.append(rewa_temp[i])

    logger.info("Discard {} dialogues which are less than 4 turn".format(count_less))
    logger.info("Discard {} state-action pairs which over length".format(count_over))

    return data_list, rewa_list


def data_process(dataset, tokenizer=None, args=None, logger=None, AtEc_train=False):
    if AtEc_train:
        train_dial = dataset["train"]["dialog"]
        train_dial.extend(dataset["test"]["dialog"])
        valid_dial = dataset["validation"]["dialog"]

        train_utte_pool = appen_utte(train_dial)
        valid_utte_pool = appen_utte(valid_dial)

        train_data = MyDataset_AtEc(train_utte_pool)
        valid_data = MyDataset_AtEc(valid_utte_pool)

        return train_data, valid_data

    else:
        # The dataset merge for dialog
        dial = dataset["train"]["dialog"]
        dial.extend(dataset["validation"]["dialog"])

        # The dataset merge for act
        acti = dataset["train"]["act"]
        acti.extend(dataset["validation"]["act"])

        # The dataset merge for emotion
        emot = dataset["train"]["emotion"]
        emot.extend(dataset["validation"]["emotion"])

        cosr = CosDis_reward(tokenizer, args, logger)
        rewa = make_reward(dial, acti, emot, cosr, logger)

        if args.cgr_enable:
            pred = Topic(args, logger)
            data_list, rewa_list = make_format(dial, rewa, args.num_state, args.max_stac_len, logger, pred)
        else:
            data_list, rewa_list = make_format(dial, rewa, args.num_state, args.max_stac_len, logger)

        dataset = MyDataset(data_list, rewa_list)

        return dataset


def tokenizer_process(batch, tokenizer, device):
    data, rewa = zip(*batch)
    arry_data = np.array(data)

    S_t  = arry_data[:,0].tolist()
    A_t  = arry_data[:,1].tolist()
    S_tt = arry_data[:,2].tolist()
    A_tt = arry_data[:,3].tolist()

    sou = tokenizer(S_t, A_t,   padding=True, return_tensors="pt").to(device)
    tar = tokenizer(S_tt, A_tt, padding=True, return_tensors="pt").to(device)
    rew = torch.tensor(rewa).to(device)

    return sou, tar, rew


def tokenizer_process_AtEc(batch, tokenizer, device):
    utte_list = batch
    utte = tokenizer(utte_list, padding=True, return_tensors="pt").to(device)

    return utte