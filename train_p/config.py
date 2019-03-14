from easydict import EasyDict

def get_config():
    conf = EasyDict()
    conf.vector_3dmm_list_for_train = './label_list.txt'
    conf.vector_3dmm_list_for_train = './'
    # conf.vector_3dmm_list_for_test = 'xxxx.txt'
    conf.image_dataset = r'F:\dataSet\train_aug_120x120'
    conf.vector_label = r'F:\dataSet\Face_vector'
    conf.label_max_min = './record_max_min.txt'
    conf.num_dims = 235
    conf.lr = 0.0008
    conf.batch_size = 96
    conf.epochs = 2
    conf.freq_validation = 1
    conf.freq_saved_model = 1
    conf.resume = True
    conf.saved_model = './saved_model/2.pth'
    return conf

if __name__ =='__main__':
    conf = get_config()
    import os
    print(os.path.join(conf.image_dataset,'dssa.jpg'))