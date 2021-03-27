import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_embed', nargs='?', default='[64,64,64,64]', help="the dim of each layer,include initial embedding")
    parser.add_argument('--batch_t', type=int, default=512, help="train batch size")
    parser.add_argument('--batch_s', type=int, default=100, help="test batch size")
    parser.add_argument('--epoch', type=int, default=500, help="Epoch")
    parser.add_argument('--neg_sample_k', type=int, default=1, help="negative sample number")
    parser.add_argument('--train_rate', type=float, default=0.8, help="the test set rate")
    parser.add_argument('--drop_rate', type=float, default=0.1, help="drop out")
    parser.add_argument('--drop_node', type=float, default=0.1, help="node drop out")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--use_tag', type=bool, default=False, help="use tag data")
    parser.add_argument('--all_user', type=bool, default=True, help="use tag data")
    parser.add_argument('--top_k', nargs='?', default='[10, 20]', help="top_K list")
    parser.add_argument('--path', type=str, default=r"C:\data\hetrec2011-movielens-2k-v2", help="file path")
    parser.add_argument('--model', type=str, default="NGCF", help="model")
    parser.add_argument('--optim', type=str, default="adam", help="optimizer")
    #parser.add_argument('--loss', type=str, default="softplus", help="loss function")
    parser.add_argument('--board', type=bool, default=True, help="tensorboardX")
    parser.add_argument('--seed', type=int, default=2020, help='random seed')

    return parser.parse_args()