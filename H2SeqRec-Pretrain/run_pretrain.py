import numpy as np
import random
import torch
from torch.utils.data import DataLoader, RandomSampler

import os
import argparse

from datasets import PretrainDataset
from trainers import PretrainTrainer
from models import H2SeqRecModel


from utils import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--output_embed_dir',
                        default='output_embed/', type=str)
    parser.add_argument('--dataset', default='AMT', type=str)

    # model args
    parser.add_argument("--model_name", default='Pretrain', type=str)

    parser.add_argument("--hidden_size", type=int, default=400,
                        help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int,
                        default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob",
                        type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float,
                        default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate of adam")
    parser.add_argument("--batch_size", type=int,
                        default=2048, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of epochs (default 200)")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # pre train args
    parser.add_argument("--pre_epochs", type=int, default=300,
                        help="number of pre_train epochs. default: 300")
    parser.add_argument("--pre_batch_size", type=int, default=100)

    parser.add_argument("--mask_p", type=float,
                        default=0.2, help="mask probability")
    parser.add_argument("--mip_weight", type=float,
                        default=0.1, help="mip loss weight")
    parser.add_argument("--sp_weight", type=float,
                        default=1.0, help="sp loss weight")
    parser.add_argument("--hlp_weight", type=float,
                        default=1.0, help="hlp loss weight")

    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float,
                        default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float,
                        default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()
    args.output_embed_dir = args.output_embed_dir + str(args.hidden_size) + '/'

    # set_seed(args.seed)
    check_path(args.output_dir)
    check_path(args.output_embed_dir)

    if args.data_dir[-1] == "/":
        args.data_dir = args.data_dir[:-1]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda" if args.cuda_condition else "cpu")

    seqs, user_map, user_negs, date_map, item_map, \
        MAX_SEQ_LEN, MAX_ITEM_ID = load_data(
            args.data_dir + "/" + args.dataset)

    args.mask_id = MAX_ITEM_ID + 1
    args.item_size = MAX_ITEM_ID + 2  # begin with 1, add a mask_id
    args.user_size = len(user_map) + 1

    # save model args
    args_str = f'{args.model_name}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(args)
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    args_str_embed = f'pretrain-{args.dataset}'
    time_info = time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())
    args.embed_file = os.path.join(
        args.output_embed_dir, args_str_embed + "."+time_info + '.pkl')

    model = H2SeqRecModel(args=args)
    trainer = PretrainTrainer(model, None, None, None, args)

    for epoch in range(args.pre_epochs):
        pretrain_dataset = PretrainDataset(
            args, seqs, user_map, user_negs, date_map, item_map)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(
            pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)

        trainer.pretrain(epoch, pretrain_dataloader)

        if (epoch+1) % 10 == 0:
            ckp = f'{args.dataset}-epochs-{epoch+1}.pt'
            checkpoint_path = os.path.join(args.output_dir, ckp)
            trainer.save(checkpoint_path)
            print(trainer)
            
    embeds = np.zeros((args.item_size, args.hidden_size))
    for i in range(0, args.item_size):
        idx = torch.LongTensor([i]).to(args.device)
        i_item = model.item_embeddings(idx)[0]
        # print(i_item.shape)
        if args.cuda_condition:
            i_item = i_item.cpu().detach().numpy()
        else:
            i_item = i_item.detach().numpy()
        embeds[i] = i_item
    pickle.dump(embeds, open(args.embed_file, 'wb'), -1)

main()
