def train():
    from lib.transfer_learn.transfer_factory import TransferFactory
    tf = TransferFactory()
    tf.run()

if __name__ == '__main__':
    train()
    