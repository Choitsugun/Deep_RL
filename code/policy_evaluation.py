from data_process_eval import data_process, Collater
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from deep_Q_net import Deep_Q_net
from hyperparams import set_args
from tqdm import tqdm
import transformers
from utils import*

def save_model(agent, epoch):
    if args.cgr_enable:
        agent.qnet.save_pretrained(args.save_model_path + "/DQN/cgr/epoch{}/".format(epoch))
        torch.save({'ff1': agent.ff1.state_dict()}, args.save_model_path + "/DQN/cgr/epoch{}/ff1.pth".format(epoch))
        torch.save({'ff2': agent.ff2.state_dict()}, args.save_model_path + "/DQN/cgr/epoch{}/ff2.pth".format(epoch))
        logger.info("Saved the coarse-grained DQN model of epoch:{}".format(epoch))
    else:
        agent.qnet.save_pretrained(args.save_model_path + "/DQN/fgr/epoch{}/".format(epoch))
        torch.save({'ff1': agent.ff1.state_dict()}, args.save_model_path + "/DQN/fgr/epoch{}/ff1.pth".format(epoch))
        torch.save({'ff2': agent.ff2.state_dict()}, args.save_model_path + "/DQN/fgr/epoch{}/ff2.pth".format(epoch))
        logger.info("Saved the fine-grained DQN model of epoch:{}".format(epoch))


def train_epoch(agent, train_dataloader, optimizer, scheduler, epoch):
    batch_step = len(train_dataloader)
    total_loss = 0

    for batch_idx, (sou, tar, rew) in enumerate(tqdm(train_dataloader)):
        try:
            loss = agent(sou=sou, tar=tar, rew=rew)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx+1) % args.inter_sync == 0:
                agent.sync_qnet()

            total_loss += float(loss)
            del loss, sou, tar, rew

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    logger.info("Training epoch:{} Loss:{}".format(epoch, total_loss/batch_step))


def train(agent, train_data, tokenizer):
    collate_fn = Collater(tokenizer, args.device)
    train_dataloader = \
    DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, collate_fn=collate_fn)

    # ========== train ========== #
    agent.train()
    t_total = len(train_dataloader) * args.epochs
    optimizer = transformers.AdamW(list(agent.qnet.parameters())+list(agent.ff1.parameters())+list(agent.ff2.parameters()), lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, args.warm_step, t_total)
    logger.info("Starting training")

    for epoch in range(1, args.epochs+1):
        train_epoch(agent, train_dataloader, optimizer, scheduler, epoch)

        if epoch >= args.after_save:
            save_model(agent, epoch)

    logger.info("Training finished")


def main():
    device_assign(args, logger)
    logger.info("Loading daily dialog dataset")

    if args.dataset_path:
        dataset = load_from_disk(dataset_path=args.dataset_path)
    else:
        dataset = load_dataset("daily_dialog", cache_dir=args.ds_cache_dir)
        dataset.save_to_disk("../dataset/")

    logger.info("Starting dataset process")
    tokenizer = BertTokenizer.from_pretrained(args.DQN_pretrained)
    train_data = data_process(dataset, tokenizer, args, logger)

    agent = Deep_Q_net(args, logger)
    agent = to_device(args.device, logger, agent)
    train(agent, train_data, tokenizer)


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    main()