import random
from numpy.core.fromnumeric import squeeze

import torch
from torch.utils.data import Dataset
from utils import mask_sample


class PretrainDataset(Dataset):

    # def __init__(self, args, seqs, user_map, user_negs, date_map, item_map, item_attr):
    def __init__(self, args, seqs, user_map, user_negs, date_map, item_map):
        self.args = args
        self.max_len = args.max_seq_length

        self.seqs = seqs
        self.user_map = user_map
        self.user_negs = user_negs
        self.date_map = date_map
        self.item_map = item_map
        # self.item_attr = item_attr

        self.uid = list(user_map.keys())

    def neg_sample(self, uid, item_set):
        l = len(self.user_neg[uid])
        neg_item = self.user_neg[uid][random.randint(0, l)]
        while neg_item in item_set:
            neg_item = self.user_neg[uid][random.randint(0, l)]
        return neg_item

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, index):
        uid = self.uid[index]

        # Mask Item with  Same Attribute
        sequence = self.seqs[uid]
        masked_seq_1, masked_seq_2 = [], []
        pos_1, pos_2 = mask_sample(sequence)
        if pos_2 != -1:
            for idx in range(len(sequence) - 1):
                item = sequence[idx]
                masked_seq_1.append(
                    self.args.mask_id if idx == pos_1 else item)
                masked_seq_2.append(
                    self.args.mask_id if idx == pos_2 else item)
        else:
            masked_seq_1 = sequence[:-1]
            masked_seq_2 = sequence[:-1]

        # add mask at the last position
        masked_seq_1.append(self.args.mask_id)
        masked_seq_2.append(self.args.mask_id)

        assert len(masked_seq_1) == len(sequence)
        assert len(masked_seq_1) == len(masked_seq_2)

        # Mask Subsequence Items
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            l = min(len(sequence), len(self.user_negs[uid]))
            sample_length = random.randint(1, l // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(
                0, len(self.user_negs[uid]) - sample_length)
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.user_negs[uid][neg_start_id:neg_start_id + sample_length]

            masked_segment_sequence = sequence[:start_id] + [self.args.mask_id] * sample_length + sequence[
                start_id + sample_length:]
            pos_segment = [self.args.mask_id] * start_id + pos_segment + [self.args.mask_id] * (
                len(sequence) - (start_id + sample_length))
            neg_segment = [self.args.mask_id] * start_id + neg_segment + [self.args.mask_id] * (
                len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # Hyperedge Link Prediction
        dates = list(self.user_map[uid].keys())
        date = random.sample(dates, 1)[0]
        hpyeredge_seq = self.user_map[uid][date]
        hyperedge_pos_seq = self.user_map[uid][date]
        hyperedge_neg_seq = self.user_map[uid][date]

        # padding sequence
        pad_len = self.max_len - len(sequence)
        sequence = [0] * pad_len + sequence
        masked_item_seq_1 = [0] * pad_len + masked_seq_1
        masked_item_seq_2 = [0] * pad_len + masked_seq_2
        masked_segment_sequence = [0] * pad_len + masked_segment_sequence
        pos_segment = [0]*pad_len + pos_segment
        neg_segment = [0]*pad_len + neg_segment

        sequence = sequence[-self.max_len:]
        masked_item_seq_1 = masked_item_seq_1[-self.max_len:]
        masked_item_seq_2 = masked_item_seq_2[-self.max_len:]
        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

        pad_len_hlp = self.max_len - len(hpyeredge_seq)
        hpyeredge_seq = [0] * pad_len_hlp + hpyeredge_seq
        hyperedge_pos_seq = [0] * pad_len_hlp + hyperedge_pos_seq
        hyperedge_neg_seq = [0] * pad_len_hlp + hyperedge_neg_seq

        hpyeredge_seq = hpyeredge_seq[-self.max_len:]
        hyperedge_pos_seq = hyperedge_pos_seq[-self.max_len:]
        hyperedge_neg_seq = hyperedge_neg_seq[-self.max_len:]

        assert len(sequence) == self.max_len
        assert len(masked_item_seq_1) == self.max_len
        assert len(masked_item_seq_2) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len
        assert len(hpyeredge_seq) == self.max_len
        assert len(hyperedge_pos_seq) == self.max_len
        assert len(hyperedge_neg_seq) == self.max_len

        cur_tensors = (torch.tensor(sequence, dtype=torch.long),
                       torch.tensor(masked_item_seq_1, dtype=torch.long),
                       torch.tensor(masked_item_seq_2, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long),
                       torch.tensor(hpyeredge_seq, dtype=torch.long),
                       torch.tensor(hyperedge_pos_seq, dtype=torch.long),
                       torch.tensor(hyperedge_neg_seq, dtype=torch.long))
        return cur_tensors
