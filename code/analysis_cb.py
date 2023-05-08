from datasets import load_dataset, load_from_disk
from data_process_acto import data_process
from deep_Q_net import Deep_Q_net
from hyperparams import set_args
from topic_pred import Topic
from tqdm import tqdm
from utils import*

def action_expect(state_data):
    logger.info('Start the behavior analysis of coarse-grained DQN')

    pred = Topic(args, logger)
    topics = Topic(args, logger, get_classmap=True).topics
    agent_cgr = Deep_Q_net(args, logger, if_train=False, graine_sele="cgr")
    agent_cgr = to_device(args.device, logger, agent_cgr)

    state_data = state_data[:args.n_anal]
    map = dict.fromkeys(topics)
    for t in topics:
        map[t] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for index, cont in enumerate(tqdm(state_data)):
        cont_topic = pred.predict(cont[-1])
        scor = agent_cgr(S_t=cont, A_t=topics, if_train=False)
        _, indices = scor.topk(1, sorted=False)
        map[cont_topic][indices.item()] = map[cont_topic][indices.item()] + 1

    print("Mapping dict:", map)


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