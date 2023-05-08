from cossi_reward_calc import CosDis_reward
from transformers import BertTokenizer
from hyperparams import set_args
import numpy as np
from utils import*
import codecs
import re

def cos_dis_cal(resp_list, tokenizer, args, logger):
    cosr = CosDis_reward(tokenizer, args, logger)
    sum_reward = 0
    sum_distan = 0
    sample_l = []

    for resp in resp_list:
        cos_reward = cosr.get_cosdis_reward(resp)
        cos_distan = 1 - cos_reward
        sum_reward = cos_reward + sum_reward
        sum_distan = cos_distan + sum_distan
        sample_l.append(cos_distan)

    mean_cosr = sum_reward / len(resp_list)
    mean_cosd = sum_distan / len(resp_list)
    std_cosd  = np.std(sample_l)

    return mean_cosr, mean_cosd, std_cosd


def resp_len_cal(resp_list):
    sum_reward = 0
    sum_reslen = 0
    sample_l = []

    for resp in resp_list:
        resp_len = len(resp.split())
        sum_reslen = sum_reslen + resp_len
        sample_l.append(resp_len)

        if resp_len < 7:
            sum_reward = sum_reward - 1
        elif resp_len < 15:
            sum_reward = sum_reward + 0
        elif resp_len < 20:
            sum_reward = sum_reward + 0.5
        else:
            sum_reward = sum_reward + 1

    mean_resr = sum_reward / len(resp_list)
    mean_resl = sum_reslen / len(resp_list)
    std_resl  = np.std(sample_l)

    return mean_resr, mean_resl, std_resl


def act_ask_cal(resp_list):
    ques_list = ["how", "why", "where", "what", "when", "who"]
    sum_act = 0

    for resp in resp_list:
        for pattern in ques_list:
            if re.match(pattern, resp.lower()) is not None \
            or re.search("\?", resp.lower()) is not None:
                sum_act = sum_act + 1
                break

    mean_actr = sum_act / len(resp_list)
    mean_actn = sum_act / len(resp_list)

    return mean_actr, mean_actn


def emo_lau_cal(resp_list):
    emot_list = ["aha ", "whoa ", "gee ", "oh ", "wow ", "ha ", "amazing ", "really ?"]
    sum_emo = 0

    for resp in resp_list:
        resp = resp.strip()
        for pattern in emot_list:
            if re.search(pattern, resp.lower()) is not None:
                sum_emo = sum_emo + 1
                break

    mean_emor = sum_emo / len(resp_list)
    mean_emon = sum_emo / len(resp_list)

    return mean_emor, mean_emon


def main():
    device_assign(args, logger)

    resp_list = []
    l_dial = [line.strip() for line in codecs.open(args.anal_path, 'r', 'utf-8').readlines() if line.strip()]

    for i, resp in enumerate(l_dial):
        if i % 2 != 0:
            resp_list.append(resp)

    tokenizer = BertTokenizer.from_pretrained(args.DQN_pretrained)
    mean_cosr, mean_cosd, std_cosd = cos_dis_cal(resp_list, tokenizer, args, logger)
    print("Mean cos reward: {}".format(mean_cosr))
    print("Mean cos distan: {}".format(mean_cosd))
    print("Cos distanc std: {}".format(std_cosd))

    mean_resr, mean_resl, std_resl = resp_len_cal(resp_list)
    print("Mean res reward: {}".format(mean_resr))
    print("Mean res lenthg: {}".format(mean_resl))
    print("Respnos len std: {}".format(std_resl))

    mean_actr, mean_actn = act_ask_cal(resp_list)
    print("Mean act reward: {}".format(mean_actr))
    print("Mean act number: {}".format(mean_actn))

    mean_emor, mean_emon = emo_lau_cal(resp_list)
    print("Mean emo reward: {}".format(mean_emor))
    print("Mean emo number: {}".format(mean_emon))


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    main()