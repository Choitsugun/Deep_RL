from torch.utils.data import Dataset
from topic_pred import Topic
from tqdm import tqdm

class Collater_GPT():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        dial = tokenizer_process_gpt(batch, self.tokenizer, self.device)
        return dial


class Collater_T5():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        cont, resp = tokenizer_process_t5(batch, self.tokenizer, self.device)
        return cont, resp


class MyDataset_GPT(Dataset):
    def __init__(self, dial):
        self.dial = dial

    def __getitem__(self, index):
        b_dial = self.dial[index]
        return b_dial

    def __len__(self):
        return len(self.dial)


class MyDataset_T5(Dataset):
    def __init__(self, cont, resp):
        self.cont = cont
        self.resp = resp

    def __getitem__(self, index):
        b_cont = self.cont[index]
        b_resp = self.resp[index]
        return b_cont, b_resp

    def __len__(self):
        return len(self.cont)


def get_state_data(dial_list, max_len, logger):
    logger.info("Starting extract the state (context) process for actor improvement")
    state_data = []
    count_over = 0

    for i, dial in enumerate(tqdm(dial_list)):
        state = dial[:-1]

        if len(" ".join(state).split()) > max_len:
            count_over = count_over + 1
        else:
            state_data.append(state)

    logger.info("Discard {} states which over length".format(count_over))

    return state_data


def make_common_data(dial, num_state, logger):
    logger.info("Starting the common data process for actor training")
    temp_list = []
    dial_list = []

    # Strip the space in beginning and end of uttes, since different form Bert, GPT and T5 tokenizer will care them
    for i, uttes in enumerate(tqdm(dial)):
        utte_list = []

        for utte in uttes:
            utte_list.append(utte.strip())

        temp_list.append(utte_list)

    # For example num_state=3, then dial_list=[[u1,u2], [u1,u2,u3], [u1,u2,u3,u4], [u2,u3,u4,u5],...]
    for i, uttes in enumerate(tqdm(temp_list)):
        u_len = len(uttes)

        for ii in range(u_len-1):
            if ii < num_state:
                dial_list.append(uttes[:ii+2])
            else:
                dial_list.append(uttes[ii+1-num_state:ii+2])

    return dial_list


def make_gpt_data(dial_list, eos_token, max_len, pred, logger):
    logger.info("Starting the data process for GPT training")
    data_list = []
    count = 0

    if pred is not None:
        max_len = max_len + 5

    # w/  topic: topic[EOS] utterance1 utterance2 utterance3[EOS] response[EOS]
    # w/o topic: utterance1 utterance2 utterance3[EOS] response[EOS]
    for index, dialog in enumerate(tqdm(dial_list)):
        cont = " ".join(dialog[:-1]) + eos_token
        resp = dialog[-1] + eos_token

        if pred is not None:
            d = pred.predict(dialog[-1]) + eos_token + " " + cont + " " + resp
        else:
            d = cont + " " + resp

        if len(d.split()) > max_len:
            count = count + 1
        else:
            data_list.append(d)

    logger.info("Discard {} dialogues which are length over".format(count))

    return data_list


def make_t5_data(dial_list, max_len, pred, logger):
    logger.info("Starting the data process for T5 training")
    cont_list = []
    resp_list = []
    count = 0

    if pred is not None:
        max_len = max_len + 20

    # w   topic cont_list: {Instruction} [CONTEXT] utterance1 EOS utterance2 EOS utterance3
    # w/o topic cont_list: [CONTEXT] utterance1 EOS utterance2 EOS utterance3
    # resp_list: response
    for index, dialog in enumerate(tqdm(dial_list)):
        resp = dialog[-1]

        if pred is not None:
            instruction = "Instruction: given a dialog context, you need to response related to {}."
            cont = instruction.format(pred.predict(dialog[-1])) + " " + "[CONTEXT]" + " " + " EOS ".join(dialog[:-1])
        else:
            cont = "[CONTEXT]" + " " + " EOS ".join(dialog[:-1])

        if len((cont+" "+resp).split()) > max_len+5:
            count = count + 1
        else:
            cont_list.append(cont)
            resp_list.append(resp)

    logger.info("Discard {} dialogues which are length over".format(count))

    return cont_list, resp_list


def data_process(args, logger, dataset=None, tokenizer=None, state_only=False, dial_list=None, if_anal=False):
    pred = None

    if dataset is not None and dial_list is None:
        if not if_anal:
            train_dial = dataset["train"]["dialog"]
            train_dial.extend(dataset["validation"]["dialog"])
            dial_list = make_common_data(train_dial, args.num_state, logger)
        else:
            test_dial = dataset["test"]["dialog"]
            dial_list = make_common_data(test_dial, args.num_state, logger)

    else:
        args.cgr_enable = False

    if state_only:
        my_data = get_state_data(dial_list, args.max_cont_len, logger)

    else:
        if args.cgr_enable:
            pred = Topic(args, logger)

        if args.actor_name == "GPT2" or args.actor_name == "DialoGPT":
            dial_list = make_gpt_data(dial_list, tokenizer.eos_token, args.max_dial_len, pred, logger)
            my_data = MyDataset_GPT(dial_list)

        if args.actor_name == "T5" or args.actor_name == "GODEL":
            cont_list, resp_list = make_t5_data(dial_list, args.max_dial_len, pred, logger)
            my_data = MyDataset_T5(cont_list, resp_list)

    return my_data


def tokenizer_process_gpt(batch, tokenizer, device):
    b_dial = batch
    dial = tokenizer(b_dial, padding=True, return_tensors="pt").to(device)

    return dial


def tokenizer_process_t5(batch, tokenizer, device):
    b_cont, b_resp = zip(*batch)
    cont = tokenizer(b_cont, padding=True, return_tensors="pt").to(device)
    resp = tokenizer(b_resp, padding=True, return_tensors="pt").to(device)

    return cont, resp,