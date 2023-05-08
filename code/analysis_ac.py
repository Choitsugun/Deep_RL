from datasets import load_dataset, load_from_disk
from data_process_acto import data_process
from hyperparams import set_args
from inference import Inference
from topic_pred import Topic
from tqdm import tqdm
from utils import*

def agent_ctrl(state_data):
    logger.info('Start the agent control testing')
    actor = Inference(args, logger)

    pred = Topic(args, logger)
    topics = Topic(args, logger, get_classmap=True).topics
    actor = to_device(args.device, logger, actor)

    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    state_data = state_data[:args.n_anal]

    for index, cont in enumerate(tqdm(state_data)):
        for i, topic in enumerate(topics):
            if args.cgr_enable:
                resps = actor.generate(cont, topic)
            else:
                resps = actor.generate(cont)

            for resp in resps:
                if topic == pred.predict(resp):
                    count[i] = count[i] + 1

    print("Correct rate: {}".format(sum(count)/len(count)/args.n_anal))
    print("Hit score: {}".format(count))


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
    agent_ctrl(state_data)


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    main()