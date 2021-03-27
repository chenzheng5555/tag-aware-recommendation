import os
import time
from .utils import printc, data_statistic, get_edge, deal_index_fuc, data_split, user_interaction, user_assignment

file_name = {
    'hetrec2011-lastfm-2k': 'user_taggedartists.dat',
    'hetrec2011-movielens-2k-v2': 'user_taggedmovies.dat',
    'hetrec2011-delicious-2k': 'user_taggedbookmarks.dat',
    #'lastfm_2k': 'user_artists.dat',
}
train_test_dataset = {
    'amazon-book',
    'gowalla',
    #'lastfm',
    'yelp2018',
}

tag_least_time = {'hetrec2011-lastfm-2k': 5, 'hetrec2011-movielens-2k-v2': 5, 'hetrec2011-delicious-2k': 15}


class DataBase():
    '''
    procedure two data: every line just has 1-one interaction,2-interactions\n
    '''
    def __init__(self, cfg):
        printc(k=50)
        start = time.time()
        base_name = os.path.basename(cfg.path)

        if base_name in train_test_dataset:
            self.train_set, max_1 = user_interaction(cfg.path, "train.txt")
            self.test_set, max_2 = user_interaction(cfg.path, "test.txt")
            self.num_user, self.num_item = [max(i, j) for i, j in zip(max_1, max_2)]
            self.edge_index, _, _ = get_edge(self.train_set, self.num_user, self.num_item)

        elif base_name in file_name.keys():
            self.user_items, self.tagging, max_node = user_assignment(cfg.path, file_name[base_name], tag_least_time[base_name], cfg.use_tag)
            self.user_items, self.tagging, max_node = deal_index_fuc(self.user_items, self.tagging)
            self.num_user, self.num_item, self.num_tag = max_node
            self.train_set, self.test_set, self.valid_set = data_split(self.user_items, cfg.train_rate, valid_rate=None)

            self.edge_index, _, _ = get_edge(self.train_set, self.num_user, self.num_item, self.tagging)

        else:
            raise NotImplementedError

        self.node_list = [self.num_user, self.num_item]
        if cfg.use_tag:
            self.node_list.append(self.num_tag)

        printc(f"database init time: {time.time()-start}")
        printc(k=50)