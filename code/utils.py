import logging
import torch
import sys
import os

def make_mask(inputs, target_value, count):
    """
    count: position of target_value to start mask
    """
    bl = (inputs == target_value)
    cumsum = torch.cumsum(bl, dim=1)
    mask = torch.where(bl, cumsum-1, cumsum)
    return (mask >= count).to(inputs.device)


def GPT_batch_buil(dail, eos_token_id, cgr_enable):
    Ge_in_ids = dail["input_ids"]
    Ge_atte_mask = dail["attention_mask"]

    if cgr_enable:
        cont_mask = make_mask(inputs=dail["input_ids"], target_value=eos_token_id, count=2)
    else:
        cont_mask = make_mask(inputs=dail["input_ids"], target_value=eos_token_id, count=1)

    Ge_labels = torch.where(cont_mask & (dail["attention_mask"]==1), dail["input_ids"], -100)

    return Ge_in_ids, Ge_atte_mask, Ge_labels


def T5_batch_buil(cont, resp, pad_token_id):
    labels = resp.input_ids
    labels[labels == pad_token_id] = -100
    input_ids, attention_mask = cont.input_ids, cont.attention_mask

    return input_ids, attention_mask, labels


def expansion_embed_init(model, nexpan, logger):
    if nexpan < 0:
        logger.info("The length of tokenizer shorter than the vocab size")
        sys.exit()
    elif nexpan == 0:
        logger.info("We don't resize the embedding tabel since the length of tokenizer equal vocab size")
    else:
        params = model.state_dict()
        embeddings = params['transformer.wte.weight']

        pre_expansion_embeddings = embeddings[:-nexpan, :]
        mu = torch.mean(pre_expansion_embeddings, dim=0)
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5*sigma)

        new_embeddings = torch.stack(tuple((dist.sample() for _ in range(nexpan))), dim=0)
        embeddings[-nexpan:, :] = new_embeddings
        params['transformer.wte.weight'] = embeddings
        model.load_state_dict(params)

        logger.info("Resize the length: {} for the embedding tabel".format(nexpan))


def create_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def device_assign(args, logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    torch.multiprocessing.set_start_method("spawn")
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device:{}".format(args.device))


def to_device(device, logger, *params):
    if len(params) == 1:
        logger.info("One model is discovered")
        model = params[0]
        model.to(device)
        logger.info("Using {} to train/eval it".format(device))

        return model

    elif len(params) == 2:
        logger.info("Two models are discovered")
        model1, model2 = params
        model1.to(device)
        model2.to(device)
        logger.info("Using {} to train/eval them".format(device))

        return model1, model2

    elif len(params) == 3:
        logger.info("Three models are discovered")
        model1, model2, model3 = params
        model1.to(device)
        model2.to(device)
        model3.to(device)
        logger.info("Using {} to train/eval them".format(device))

        return model1, model2, model3

    else:
        logger.info("Invalid number of models, please check the argument")
        sys.exit()


"""
def begin_suppress(resp, begi_list):
    sptk_list = []
    begin_token = resp[0].split()[0]
    begi_list.append(begin_token)
    begi_dict = dict(Counter(begi_list))

    for token in begi_list:
        if begi_dict[token]/len(begi_list) >= args.repet_suppr:
            sptk_list.append(token)

    return sptk_list, begi_list
"""