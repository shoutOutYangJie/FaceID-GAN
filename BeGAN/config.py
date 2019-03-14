from easydict import EasyDict

def get_config():
    conf = EasyDict()
    conf.epoch = 50
    conf.batch_size = 32
    conf.save_dir = './saved_model'
    conf.result_dir = './result_model'
    conf.dataset = 'CASIA'
    conf.log_dir = './saved_model'
    conf.gpu_mode = True
    conf.gan_type = 'began'
    conf.input_size = 128
    conf.beta1 = 0.9
    conf.beta2 = 0.999
    conf.repeat_num = 5
    conf.z_dim = 64
    return conf