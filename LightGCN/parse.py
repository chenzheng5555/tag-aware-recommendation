import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--train_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--latent_dim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layers', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--neg_sample_k', type=int,default=1,
                        help="the weight decay for l2 normalizaton")

    parser.add_argument('--test_batch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--path', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon]")
    

    parser.add_argument('--topks', nargs='?',default="[20,100]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--epochs', type=int,default=1000)

    parser.add_argument('--seed', type=int, default=2020, help='random seed')

    return parser.parse_args()