from analysis_sp import cos_dis_cal, resp_len_cal, act_ask_cal, emo_lau_cal
from datasets import load_dataset, load_from_disk
from data_process_acto import data_process
from transformers import BertTokenizer
from hyperparams import set_args
from inference import Inference
from tqdm import tqdm
from utils import*
import codecs

def testing(state_data):
    logger.info('Start the response generation for testing set')
    actor = Inference(args, logger, if_interact=True).to(args.device)
    actor = to_device(args.device, logger, actor)
    state_data = state_data[:args.n_anal]
    resp_list = []

    if args.gene_resu:
        if not os.path.exists(args.resu_path):
            os.makedirs(args.resu_path)
        file = codecs.open(os.path.join(args.resu_path, "result"), 'w', 'utf8')

    for index, cont in enumerate(tqdm(state_data)):
        resp = actor.generate(cont)
        resp_list.extend(resp)

        if args.gene_resu:
            file.write("cont:" + str(cont) + "\n")
            file.write("resp:" + str(resp) + "\n\n")

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


def main():
    device_assign(args, logger)

    logger.info("Loading daily dialog dataset")
    if args.dataset_path:
        dataset = load_from_disk(dataset_path=args.dataset_path)
    else:
        dataset = load_dataset("daily_dialog", cache_dir=args.ds_cache_dir)
        dataset.save_to_disk("../dataset/")

    logger.info("Starting dataset process")
    state_data = data_process(args, logger, dataset=dataset, state_only=True, if_anal=True)
    testing(state_data)


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    main()