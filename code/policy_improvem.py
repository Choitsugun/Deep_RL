from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from data_process_acto import data_process
from deep_Q_net import Deep_Q_net
from hyperparams import set_args
from inference import Inference
from train_actor import train
from topic_pred import Topic
from tqdm import tqdm
from utils import*
import pickle

def genation_save(s2d_data=None):
    logger.info('Start the response sampling and use DQN to score them')
    actor = Inference(args, logger)

    if s2d_data is not None:
        if args.cgr_enable:
            topics = Topic(args, logger, get_classmap=True).topics
            agent_cgr = Deep_Q_net(args, logger, if_train=False, graine_sele="cgr")
            agent_fgr = Deep_Q_net(args, logger, if_train=False, graine_sele="fgr")
            actor, agent_cgr, agent_fgr = to_device(args.device, logger, actor, agent_cgr, agent_fgr)
        else:
            agent_fgr = Deep_Q_net(args, logger, if_train=False, graine_sele="fgr")
            actor, agent_fgr = to_device(args.device, logger, actor, agent_fgr)

        for index, cont in enumerate(tqdm(s2d_data)):
            if args.cgr_enable:
                scor = agent_cgr(S_t=cont, A_t=topics, if_train=False)
                _, indices = scor.topk(1, sorted=False)
                resp = actor.generate(cont, topics[indices.item()])
                scor = agent_fgr(S_t=cont, A_t=resp, if_train=False)
                _, indices = scor.topk(1, sorted=False)
            else:
                resp = actor.generate(cont)
                scor = agent_fgr(S_t=cont, A_t=resp, if_train=False)
                _, indices = scor.topk(1, sorted=False)

            # Updata the dataset
            cont.append(resp[indices.item()].strip())

        if not os.path.exists(args.imprdt_path):
            os.makedirs(args.imprdt_path)

        logger.info("Saving the improved dialogue data")
        f = open(args.imprdt_path+"/improved_data.pth", "wb")
        pickle.dump(s2d_data, f)
        f.close()
    else:
        logger.info("Restore the improved dialogue from saved data")
        f = open(args.imprdt_path+"/improved_data.pth", "rb")
        s2d_data = pickle.load(f)

    return s2d_data


def main():
    device_assign(args, logger)

    if not args.from_imprdt:
        logger.info("Loading daily dialog dataset")
        if args.dataset_path:
            dataset = load_from_disk(dataset_path=args.dataset_path)
        else:
            dataset = load_dataset("daily_dialog", cache_dir=args.ds_cache_dir)
            dataset.save_to_disk("../dataset/")

        logger.info("Starting dataset process")
        state_data = data_process(args, logger, dataset=dataset, state_only=True)
        dial_list = genation_save(s2d_data=state_data)
    else:
        dial_list = genation_save()

    if args.actor_name == "GPT2" or args.actor_name == "DialoGPT":
        actor = AutoModelForCausalLM.from_pretrained(args.Forimpr_checkp.format(args.actor_name))
        tokenizer = AutoTokenizer.from_pretrained(args.Forimpr_pretrained.format(args.actor_name))
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Restored the {} model from the check point".format(args.actor_name))

    elif args.actor_name == "T5" or args.actor_name == "GODEL":
        actor = AutoModelForSeq2SeqLM.from_pretrained(args.Forimpr_checkp.format(args.actor_name))
        tokenizer = AutoTokenizer.from_pretrained(args.Forimpr_pretrained.format(args.actor_name))
        logger.info("Restored the {} model from the check point".format(args.actor_name))

    else:
        logger.info("Please reset the model name!")
        sys.exit()

    actor = to_device(args.device, logger, actor)
    impro_data = data_process(args, logger, tokenizer=tokenizer, dial_list=dial_list)
    train(actor, impro_data, tokenizer, args, logger, improve=True)


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    main()