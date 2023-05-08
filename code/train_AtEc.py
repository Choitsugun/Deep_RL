from data_process_eval import data_process, Collater_AtEc
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from autoencoder import AutoEncoder
from hyperparams import set_args
from tqdm import tqdm
import transformers
from utils import*

def save_model(RewM, epoch):
    save_path = args.save_model_path + "/AtEc/epoch{}".format(epoch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'AtEc': RewM.state_dict()}, save_path + "/AtEc.pth")

    logger.info("Saved the AtEc model at epoch:{}".format(epoch))


def train_epoch(RewM, train_dataloader, optimizer, scheduler, epoch):
    RewM.train()
    total_loss = 0
    batch_step = len(train_dataloader)

    for batch_idx, utte in enumerate(tqdm(train_dataloader)):
        try:
            loss = RewM.forward(utte)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += float(loss)

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    nll = total_loss / batch_step
    logger.info("Training epoch:{} Loss:{}".format(epoch, nll))


def valid_epoch(RewM, valid_dataloader, epoch):
    RewM.eval()
    total_loss = 0
    batch_step = len(valid_dataloader)

    for batch_idx, batch_data in enumerate(tqdm(valid_dataloader)):
        try:
            seq = batch_data
            loss = RewM.forward(seq)
            total_loss += float(loss)

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    nll = total_loss / batch_step
    logger.info("Validating epoch:{} Loss:{}".format(epoch, nll))

    return nll


def train(RewM, train_data, valid_data, tokenizer):
    patience = 0
    best_val_loss = float('Inf')

    collate_fn = Collater_AtEc(tokenizer, args.device)

    train_dataloader = \
    DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, collate_fn=collate_fn)
    valid_dataloader = \
    DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, collate_fn=collate_fn)

    if args.forwa_only:
        # ========== eval ========== #
        logger.info("Starting validating")
        valid_epoch(RewM, valid_dataloader, None)
        logger.info("Validating finished")
    else:
        # ========== train ========== #
        t_total = len(train_dataloader) * args.epochs
        optimizer = transformers.AdamW(list(RewM.parameters()), lr=args.lr)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, args.warm_step, t_total)
        logger.info("Starting training")

        for epoch in range(1, args.epochs+1):
            train_epoch(RewM, train_dataloader, optimizer, scheduler, epoch)
            val_loss = valid_epoch(RewM, valid_dataloader, epoch)

            if val_loss < best_val_loss:
                # Save AtEc model
                save_model(RewM, epoch)
                best_val_loss = val_loss
                patience = 0
            else:
                patience = patience + 1

            if args.patience < patience:
                logger.info("Early stop due to run out of patience")
                break

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
    train_data, valid_data = data_process(dataset, AtEc_train=True)
    tokenizer = BertTokenizer.from_pretrained(args.DQN_pretrained)

    if args.forwa_only:
        RewM = AutoEncoder(args.ate_d_model, len(tokenizer), tokenizer.pad_token_id, args.device)
        RewM.load_state_dict(torch.load(args.AtEc_checkp)["AtEc"])
        logger.info("Restored the AutoEncoder model from the check point")
    else:
        RewM = AutoEncoder(args.dime_model, len(tokenizer), tokenizer.pad_token_id, args.device)
        logger.info("Initialized the AutoEncoder model")

    RewM = to_device(args.device, logger, RewM)
    train(RewM, train_data, valid_data, tokenizer)


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    main()