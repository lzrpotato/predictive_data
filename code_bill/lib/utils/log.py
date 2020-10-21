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
        fmt='%(asctime)s - %(levelname)s - %(module)s - Ln%(lineno)d - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    fhandler = logging.FileHandler(filename=logfile)
    fhandler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(fhandler)
    return logger
