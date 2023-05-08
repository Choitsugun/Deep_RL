from transformers import AutoModelForSequenceClassification, RobertaTokenizer, AutoConfig
import torch.nn.functional as F
import torch.nn as nn
import torch

class Topic(nn.Module):
    def __init__(self, args, logger, get_classmap=False):
        super().__init__()

        if not get_classmap:
            self.device = args.device
            self.model = AutoModelForSequenceClassification.from_pretrained(args.Roberta_pretrained, problem_type="multi_label_classification")
            self.tokenizer = RobertaTokenizer.from_pretrained(args.Roberta_pretrained)
            self.model.eval().to(args.device)
            self.class_mapping = self.model.config.id2label
            self.penal = F.one_hot(torch.tensor(3), num_classes=19).to(args.device) * args.topi_penal
            logger.info("Restored the Roberta model from the pretrained weight")
        else:
            config = AutoConfig.from_pretrained(args.Roberta_pretrained, problem_type="multi_label_classification")
            class_mapping = config.id2label
            self.topics = list(class_mapping.values())
            logger.info("Get the topic list: {}".format(self.topics))

    def predict(self, text:str):
        with torch.no_grad():
            tokens = self.tokenizer(text, return_tensors='pt').to(self.device)
            output = self.model(**tokens)

            probs = output[0][0].sigmoid()
            _, indices = (probs - self.penal).topk(1, sorted=False)
            topic = self.class_mapping[indices.item()]

        return topic