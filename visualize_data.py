from train import *
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import re
import nltk
from nltk.corpus import wordnet
from random import shuffle
from gensim.models import word2vec, keyedvectors
import seaborn as sns; sns.set()
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

def norm(vec):
    return vec/np.linalg.norm(vec)

def main(config,models_config, preprocessing_config, args):

    resume = args.resume

    train_logger = Logger()
    model_name = config['model_name']
    logging.info("Visualizing {}".format(model_name))

    model_config = models_config['models'][model_name]
    vocab_file_path = preprocessing_config['vocab_file'] #''datasets/processed/vocab.h5'
    labels_file_path = preprocessing_config['labels_file']
    vocab_file = h5py.File(vocab_file_path, 'r', libver='latest', swmr=True)
    labels_file = h5py.File(labels_file_path, 'r', libver='latest', swmr=True)

    data_loader = TweetsDataLoader(None,
                             sampler=None,
                             batch_size=config['data_loader']['batch_size'],
                             num_workers=config['data_loader']['num_workers'],
                                   model_config = model_config)

    model = eval(model_name)(len(vocab_file), len(labels_file), config, models_config['models'][model_name], preprocessing_config).cuda()

    trainer = Trainer(model, loss, None,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=None,
                      train_logger=train_logger)
    model = trainer.model


    # word_to_ix = {}
    embeddings = []

    # limit = math.inf
    # limit = None #50
    # print("Collecting")
    vocab = Vocabulary(vocab_file = vocab_file_path)
    word_to_ix = vocab.word_to_ix

    for ds in word_to_ix.keys():
        vec = model.word_embeddings.weight.data[word_to_ix[ds]]
        embeddings.append((ds, vec, np.linalg.norm(vec)))
    # Original alternative
    # with h5py.File(vocab_file, 'r') as h5f:
    #     print("Vocab lines: {}".format(len(h5f.keys())))
    #     for i,ds in enumerate(h5f.keys()):
    #         # if i > limit:
    #         #     break
    #         word_to_ix[ds] = int(h5f[ds].value)
    #
    #         vec = model.word_embeddings.weight.data[word_to_ix[ds]]
    #         embeddings.append((ds,vec,np.linalg.norm(vec)))

    # embeddings = list(filter(lambda x: len(wordnet.synsets(x[0])) > 0 ,embeddings))
    # shuffle(embeddings)
    embeddings.sort(key=lambda x: x[2], reverse=True)

    labels = [emb[0] for emb in embeddings]
    tokens = [emb[1] for emb in embeddings]


    # Test most similar
    most_similar_vec = []
    most_similar_labels = []
    tokens_d = [np.array(emb[1]) for emb in embeddings]
    a = keyedvectors.WordEmbeddingsKeyedVectors(len(tokens[0]))
    a.add(labels, tokens_d)

    #Test linear operations
    # test on all w2v
    try:
        tests = [a.get_vector('love')-a.get_vector('hate')+a.get_vector('scared'),
                 a.get_vector('happy')-a.get_vector('sad')+a.get_vector('crying'),
                 a.get_vector('man')-a.get_vector('woman')+a.get_vector('girl')]
        for i,test in enumerate(tests):
            print("{}: {}".format(i,a.similar_by_vector(test)))
    except Exception as e:
        logger.error(e)


    # Only works for SVME
    if config['model_name'] == "SVME":
        # test only labels w2v
        b = keyedvectors.WordEmbeddingsKeyedVectors(len(tokens[0]))

        for i, emotion in enumerate(idx2emoji.keys()):
            b.add(emotion, model.reverse(i)[0])

        tests = [norm(b.get_vector('smiling')) - norm(b.get_vector('sad_face')) + norm(b.get_vector('crying_face')),
                 b.get_vector('smiling') - b.get_vector('sad_face') + b.get_vector('crying_face'),
                 b.get_vector('devil_face') - b.get_vector('angel') + b.get_vector('inlove'),
                 b.get_vector('devil_face') - b.get_vector('angel') + b.get_vector('mad_face')
                 ]
        for i, test in enumerate(tests):
            print("{}: {}".format(i, b.similar_by_vector(test)))


        for i, emotion in enumerate(idx2emoji.keys()):
            best_embeddings, _ = model.reverse(i)
            # name = "<<{}>>".format(emotion)
            # a.add(name, best_embeddings)
            # print(a.get_vector(name))
            most_similar = a.similar_by_vector(best_embeddings, topn=3)
            print("Most similar to {}: {}".format(emotion, most_similar))
            first = True
            for ms in most_similar:
                label = ms[0]
                vec = a.get_vector(label)

                #We mark the most similar word by the class of emoji
                if first:
                    first = False
                    most_similar_labels.append("<{}>:{}".format(emotion,label))
                else:
                    most_similar_labels.append(label)
                most_similar_vec.append(vec)

            # most_similar_labels.append(idx2emoji[emotion].split("|")[0])
            # most_similar_labels.append("<<{}>>".format(emotion))
            # most_similar_vec.append(best_embeddings)
        tsne_plot_seaborn(most_similar_vec,most_similar_labels)

        # plot only imagined vectors
        imagined_vecs =[]
        imagined_labels =[]
        for i,emotion in enumerate(idx2emoji.keys()):
            best_embeddings, _ = model.reverse(i)
            imagined_vecs.append(best_embeddings)
            imagined_labels.append(emotion)


        tsne_plot_seaborn(imagined_vecs, imagined_labels)

    limit = 100

    if limit:
        tokens = tokens[:limit]
        labels = labels[:limit]

    # tsne_plot(tokens,labels)
    tsne_plot_seaborn(tokens,labels)


def tsne_plot_seaborn(tokens, labels):
    "Creates and TSNE model and plots it"

    tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5500, random_state=23)

    tokens = np.vstack(tokens)

    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    df = pd.DataFrame({'x': x, 'y':y, 'group':labels})

    p=sns.scatterplot(x='x',y='y', data = df)#,fit_reg = False)
    # add annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p.annotate(labels[line],xy=(df.x[line], df.y[line]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        # p.text(df.x[line], df.y[line]+2, df.group[line], horizontalalignment='center', size='medium', color='black')

    plt.show()
def tsne_plot(tokens, labels):
    "Creates and TSNE model and plots it"

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5500, random_state=23)

    tokens = np.vstack(tokens)

    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])


    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


if __name__ == '__main__':

    config, models_config, preprocessing_config, test_config, args = parse_config()
    main(config,models_config,preprocessing_config, args)