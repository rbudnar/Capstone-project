from keras import backend as K
from keras.callbacks import TensorBoard
## https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)
    
    def on_batch_end(self, epoch, logs=None):
        logs.update({'cust lr': K.eval(self.model.optimizer.lr)})
        super().on_batch_end(epoch, logs)    

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'cust lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)