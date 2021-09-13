from utility.word import CFG


class Abstract_training_data:
    '''get all training data, then yeild mini batch data'''
    def __init__(self, args = None) -> None:
        self.device = CFG['device']
        self.cpu_core = CFG["cpu_core"]
        self.all_train_data = None

    def get_all_training_data(self):
        raise NotImplementedError

    def reset(self):
        self.all_train_data = self.get_all_training_data()

    def mini_batch(self):
        '''最后的一个batch，如果数量少，则加入前一个batch'''
        for i in range(0, self.all_train_data.shape[0], self.batch_size):
            if i + 2 * self.batch_size > self.all_train_data.shape[0]:
                yield self.all_train_data[i:]
            else:
                yield self.all_train_data[i:i + self.batch_size]