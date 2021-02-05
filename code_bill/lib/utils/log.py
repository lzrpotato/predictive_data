import logging
import os


def setup_custom_logger(name, filename):
    filename = os.path.basename(filename)
    if not os.path.isdir('./logging'):
        os.makedirs('./logging/')

    import datetime
    current_time = datetime.datetime.now()
    t = current_time.strftime('%Y-%m-%d-%H-%M-%S')
    logfile = './logging/'+filename+'_'+t+'.log'

    formatter = logging.Formatter(
        fmt='P:%(process)d-T:%(thread)d-%(asctime)s-%(levelname)s-%(filename)s-l%(lineno)d- %(message)s',
        datefmt='%m/%d-%H:%M:%S'
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    fhandler = logging.FileHandler(filename=logfile)
    fhandler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(fhandler)
    return logger
