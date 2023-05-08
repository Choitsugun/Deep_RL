import torch.nn as nn
import torch

class EncoderRNN(nn.Module):
    def __init__(self, embed_table, hidden_size):
        super().__init__()
        self.embed_table = embed_table
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, seq, hidden=None):
        input_embeds = self.embed_table(seq["input_ids"])
        lengths = torch.sum(seq["attention_mask"], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(input_embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed, hidden)

        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, embed_table, hidden_size, vocab_size):
        super().__init__()
        self.embed_table = embed_table
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, seq, last_hidden):
        input_embeds = self.embed_table(seq["input_ids"])
        lengths = torch.sum(seq["attention_mask"], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(input_embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed, last_hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        logits = self.out(output)  # N Len V

        return logits


class AutoEncoder(nn.Module):
    def __init__(self, d_model, vocab_size, pad_id, device):
        super().__init__()
        embed_table = nn.Embedding(vocab_size, d_model)
        self.encoder = EncoderRNN(embed_table, d_model)
        self.decoder = DecoderRNN(embed_table, d_model, vocab_size)
        self.loss_fct = nn.CrossEntropyLoss()
        self.pad_id = pad_id
        self.device = device

    def forward(self, seq, if_train=True):
        hidden = self.encoder(seq)
        if if_train:
            logits = self.decoder(seq, hidden)
            labels = torch.where(seq["input_ids"] == self.pad_id, torch.tensor(-100).to(self.device), seq["input_ids"])

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return loss
        else:
            return hidden.squeeze(0)    # 1 N C → N C