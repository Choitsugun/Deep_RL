from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn as nn
from utils import*

class Inference(nn.Module):
    def __init__(self, args, logger, if_interact=False):
        super().__init__()

        if if_interact:
            if args.actor_name == "GPT2" or args.actor_name == "DialoGPT":
                actor = AutoModelForCausalLM.from_pretrained(args.Forinte_checkp.format(args.actor_name))
                tokenizer = AutoTokenizer.from_pretrained(args.Forinte_pretrained.format(args.actor_name))
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Restored the {} model from the check point".format(args.actor_name))

            elif args.actor_name == "T5" or args.actor_name == "GODEL":
                actor = AutoModelForSeq2SeqLM.from_pretrained(args.Forinte_checkp.format(args.actor_name))
                tokenizer = AutoTokenizer.from_pretrained(args.Forinte_pretrained.format(args.actor_name))
                logger.info("Restored the {} model from the check point".format(args.actor_name))

            else:
                logger.info("Please reset the model name!")
                sys.exit()

        else:
            if args.cgr_enable:
                tp_info = "w_tpi"
            else:
                tp_info = "wo_tp"

            if args.actor_name == "GPT2":
                actor = AutoModelForCausalLM.from_pretrained(args.GPT2_checkp.format(tp_info))
                tokenizer = AutoTokenizer.from_pretrained(args.GPT2_pretrained)
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Restored the GPT2 model from the check point")

            elif args.actor_name == "DialoGPT":
                actor = AutoModelForCausalLM.from_pretrained(args.DialoGPT_checkp.format(tp_info))
                tokenizer = AutoTokenizer.from_pretrained(args.DialoGPT_pretrained)
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Restored the DialoGPT model from the check point")

            elif args.actor_name == "T5":
                actor = AutoModelForSeq2SeqLM.from_pretrained(args.T5_checkp.format(tp_info))
                tokenizer = AutoTokenizer.from_pretrained(args.T5_pretrained)
                logger.info("Restored the T5 model from the check point")

            elif args.actor_name == "GODEL":
                actor = AutoModelForSeq2SeqLM.from_pretrained(args.GODEL_checkp.format(tp_info))
                tokenizer = AutoTokenizer.from_pretrained(args.GODEL_pretrained)
                logger.info("Restored the GODEL model from the check point")

            else:
                logger.info("Please reset the model name!")
                sys.exit()

        self.args = args
        self.actor = actor
        self.tokenizer = tokenizer
        self.actor.eval()

    def generate(self, message:list, topic=None):
        if self.args.actor_name == "GPT2" or self.args.actor_name == "DialoGPT":
            if topic is not None:
                cont = topic + self.tokenizer.eos_token + " " + " ".join(message) + self.tokenizer.eos_token
            else:
                cont = " ".join(message) + self.tokenizer.eos_token

            cont_enco = self.tokenizer(cont, padding=False, return_tensors="pt").to(self.args.device)
            outputs = self.actor.generate(**cont_enco, max_new_tokens=self.args.resp_ge_len,
                                            num_return_sequences=self.args.n_retur_seq,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            temperature=self.args.temperature,
                                            do_sample=True)

            resp = self.tokenizer.batch_decode(outputs[:, cont_enco.input_ids.shape[-1]:], skip_special_tokens=True)

        if self.args.actor_name == "T5" or self.args.actor_name == "GODEL":
            if topic is not None:
                instruction = "Instruction: given a dialog context, you need to response related to {}."
                cont = instruction.format(topic) + " " + "[CONTEXT]" + " " + " EOS ".join(message)
            else:
                cont = "[CONTEXT]" + " " + " EOS ".join(message)

            cont_enco = self.tokenizer(cont, padding=False, return_tensors="pt").to(self.args.device)
            outputs = self.actor.generate(**cont_enco, max_new_tokens=self.args.resp_ge_len,
                                          num_return_sequences=self.args.n_retur_seq,
                                          temperature=self.args.temperature,
                                          do_sample=True)

            resp = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return resp