from datasets import load_dataset, load_from_disk
from data_process_acto import data_process
from deep_Q_net import Deep_Q_net
from hyperparams import set_args
from inference import Inference
from topic_pred import Topic
from tqdm import tqdm
from utils import*

def action_expect(state_data):
    logger.info('Start the response sampling and use DQN to score them')
    actor = Inference(args, logger)

    topics = Topic(args, logger, get_classmap=True).topics
    agent_cgr = Deep_Q_net(args, logger, if_train=False, graine_sele="cgr")
    agent_fgr = Deep_Q_net(args, logger, if_train=False, graine_sele="fgr")
    actor, agent_cgr, agent_fgr = to_device(args.device, logger, actor, agent_cgr, agent_fgr)

    state_data = state_data[:args.n_anal]
    scor_l = []

    for index, cont in enumerate(tqdm(state_data)):
        if args.cgr_enable:
            scor = agent_cgr(S_t=cont, A_t=topics, if_train=False)
            _, indices = scor.topk(1, sorted=False)
            resp = actor.generate(cont, topics[indices])
            scor = agent_fgr(S_t=cont, A_t=resp, if_train=False)
            scor_l.extend(scor.tolist())
        else:
            resp = actor.generate(cont)
            scor = agent_fgr(S_t=cont, A_t=resp, if_train=False)
            scor_l.extend(scor.tolist())

    print("Mean Q-value: {}".format(sum(scor_l)/len(scor_l)))


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
    action_expect(state_data)


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    main()