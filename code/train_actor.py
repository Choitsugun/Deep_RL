from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from data_process_acto import data_process, Collater_GPT, Collater_T5
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from hyperparams import set_args
from tqdm import tqdm
import transformers
from utils import*

def save_model(actor, epoch, args, logger, improve):
    if not improve:
        if args.cgr_enable:
            actor.save_pretrained(args.save_model_path + "/actor/w_tpi/{}/epoch{}/".format(args.actor_name, epoch))
            logger.info("Saved the {} model of epoch:{}".format(args.actor_name, epoch))
        else:
            actor.save_pretrained(args.save_model_path + "/actor/wo_tp/{}/epoch{}/".format(args.actor_name, epoch))
            logger.info("Saved the {} model of epoch:{}".format(args.actor_name, epoch))
    else:
        actor.save_pretrained(args.save_model_path + "/actor_im/{}/epoch{}/".format(args.actor_name, epoch))
        logger.info("Saved the improved {} model of epoch:{}, by our method".format(args.actor_name, epoch))


def train_epoch(actor, train_dataloader, tokenizer, optimizer, scheduler, epoch, args, logger):
    total_ge_l = 0
    batch_step = len(train_dataloader)

    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        try:
            if args.actor_name == "GPT2" or args.actor_name == "DialoGPT":
                dail = batch_data
                d_ii, d_am, d_la = GPT_batch_buil(dail, tokenizer.eos_token_id, args.cgr_enable)
                outputs = actor(input_ids=d_ii, attention_mask=d_am, labels=d_la)

            if args.actor_name == "T5" or args.actor_name == "GODEL":
                cont, resp = batch_data
                c_ii, c_am, r_la = T5_batch_buil(cont, resp, tokenizer.pad_token_id)
                outputs = actor(input_ids=c_ii, attention_mask=c_am, labels=r_la)

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_ge_l += float(loss)

            del outputs, loss

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    ge_l = total_ge_l / batch_step
    logger.info("train_{} loss:{}".format(epoch, ge_l))


def train(actor, train_data, tokenizer, args, logger, improve=False):
    if args.actor_name == "GPT2" or args.actor_name == "DialoGPT":
        collate_fn = Collater_GPT(tokenizer, args.device)

    if args.actor_name == "T5" or args.actor_name == "GODEL":
        collate_fn = Collater_T5(tokenizer, args.device)

    train_dataloader = \
    DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, collate_fn=collate_fn)

    # ========== train ========== #
    actor.train()
    t_total = len(train_dataloader) * args.epochs
    optimizer = transformers.AdamW(list(actor.parameters()), lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, args.warm_step, t_total)
    logger.info("Starting training")

    for epoch in range(1, args.epochs+1):
        train_epoch(actor, train_dataloader, tokenizer, optimizer, scheduler, epoch, args, logger)

        if epoch >= args.after_save:
            save_model(actor, epoch, args, logger, improve)

    logger.info("Training finished")


def main():
    args = set_args()
    logger = create_logger(args)
    device_assign(args, logger)

    if args.actor_name == "GPT2":
        actor = AutoModelForCausalLM.from_pretrained(args.GPT2_pretrained)
        tokenizer = AutoTokenizer.from_pretrained(args.GPT2_pretrained)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Restored the GPT2 model from the pretrained weight")

    elif args.actor_name == "DialoGPT":
        actor = AutoModelForCausalLM.from_pretrained(args.DialoGPT_pretrained)
        tokenizer = AutoTokenizer.from_pretrained(args.DialoGPT_pretrained)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Restored the DialoGPT model from the pretrained weight")

    elif args.actor_name == "T5":
        actor = AutoModelForSeq2SeqLM.from_pretrained(args.T5_pretrained)
        tokenizer = AutoTokenizer.from_pretrained(args.T5_pretrained)
        logger.info("Restored the T5 model from the pretrained weight")

    elif args.actor_name == "GODEL":
        actor = AutoModelForSeq2SeqLM.from_pretrained(args.GODEL_pretrained)
        tokenizer = AutoTokenizer.from_pretrained(args.GODEL_pretrained)
        logger.info("Restored the GODEL model from the pretrained weight")

    else:
        logger.info("Please reset the model name!")
        sys.exit()

    logger.info("Loading daily dialog dataset")
    if args.dataset_path:
        dataset = load_from_disk(dataset_path=args.dataset_path)
    else:
        dataset = load_dataset("daily_dialog", cache_dir=args.ds_cache_dir)
        dataset.save_to_disk("../dataset/")

    logger.info("Starting dataset process")
    train_data = data_process(args, logger, dataset=dataset, tokenizer=tokenizer)
    actor = to_device(args.device, logger, actor)
    train(actor, train_data, tokenizer, args, logger)


if __name__ == '__main__':
    main()