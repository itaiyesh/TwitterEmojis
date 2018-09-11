from datasets import HDF5Dataset
from preprocessing import *
from model import *
if __name__ == '__main__':

    #TODO: This entire thing just to get features_len and num_labels - Put that in the config instead!!
    # tweets_path = 'datasets/processed/tweets.h5'
    vocab_path = 'datasets/processed/vocab.h5'
    labels_path = 'datasets/processed/labels.h5'
    #
    # dataset = HDF5Dataset(tweets_path)

    parser = argparse.ArgumentParser(description='Directory of model execution')
    parser.add_argument('-d', '--dir', default=None, type=str,
                        help='Directory of model execution')

    args = parser.parse_args()


    config_path = os.path.join(args.dir, 'config.json')
    config = json.load(open(config_path))

    path = os.path.join(config['trainer']['save_dir'], config['name'])


    vocab_file = h5py.File(vocab_path, 'r', libver='latest', swmr=True)
    labels_file = h5py.File(labels_path, 'r', libver='latest', swmr=True)

    model = LSTM(len(vocab_file), len(labels_file), config, config["models"]['LSTM'])

    checkpoint_path = os.path.join(args.dir, 'model_best.pth.tar')
    checkpoint = torch.load(checkpoint_path)
    print("loading {}".format(checkpoint_path))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.cuda()
    model.summary()

    #TODO: Save vocab must be relevant to model and execution weights!!!
    vocab_file = 'datasets/processed/vocab.h5'
    word_to_ix = {}
    with h5py.File(vocab_file, 'r') as h5f:
        for ds in h5f.keys():
            word_to_ix[ds] = int(h5f[ds].value)

    raw_text = None
    while raw_text != 'bye':
        # raw_text = input('Text:')
        raw_text = "OMG"
        text = clean(raw_text)

        # model.batch_size = 1
        # v = torch.from_numpy(make_bow_vector(text, word_to_ix)).long()
        # v = Variable([make_bow_vector(text, word_to_ix)])
        v = np.repeat([make_bow_vector(text, word_to_ix).numpy()], model.batch_size, axis=0)
        v = torch.from_numpy(v).long()#.cpu()
        output = model(v)
        # model.batch_size = 2
        # vec = make_seq_vector(text, word_to_ix, 30, 2341)#.cuda()
        # test_input = torch.LongTensor(np.repeat([vec.numpy()],32,axis = 0)).cuda()
        output = model(output)

        array = output.data.numpy()
        emotion = list(index_to_emotion.keys())[np.argmax(array)]
        print(emotion)
