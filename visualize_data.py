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
from bokeh.plotting import figure, show, output_file
from bokeh.models import Range1d, GeoJSONDataSource
from sklearn.preprocessing import MinMaxScaler
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label
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

    model.eval()

    # word_to_ix = {}
    embeddings = []

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

    tf_projector_plot('vector.tsv', 'label.tsv', tokens[:1000], labels[:1000])
    # exit(0)

    # Test most similar
    most_similar_vec = []
    most_similar_labels = []
    tokens_d = [np.array(emb[1]) for emb in embeddings]
    a = keyedvectors.WordEmbeddingsKeyedVectors(len(tokens[0]))
    a.add(labels, tokens_d)

    #Test linear operations
    # test on all w2v
    try:
        tests = [['love','hate','scared'],
                 ['happy','sad','crying'],
                 ['man','woman','girl']]

        for i,test in enumerate(tests):
            print("{} is to {} as {} is to {}".format(test[0], test[1], a.most_similar(positive=test[:2], negative=test[2:], topn=3),test[2]))
    except Exception as e:
        logger.error(e)


    # Only works for SVME
    if config['model_name'] == "SVME":
        # test only labels w2v
        b = keyedvectors.WordEmbeddingsKeyedVectors(len(tokens[0]))

        for i, emotion in enumerate(idx2emoji.keys()):
            b.add(emotion, model.reverse(i)[0])

        tests = [['inlove', 'kissing', 'sick_face'],
                 ['heart', 'inlove', 'straight_face'],
                 ['devil_face', 'angel', 'inlove'],
                 ['sad_face','crying_face', 'smiling'],
                 ['inlove', 'liplicking', 'sad_face']]

        for i, test in enumerate(tests):
            print("{} is to {} as {} is to {}".format(test[0], test[1],
                                                      b.most_similar(positive=test[:2], negative=test[2:], topn=3), test[2]))

        emojis = []
        emoji_vecs = []
        for i, emotion in enumerate(idx2emoji.keys()):
            best_embeddings, _ = model.reverse(i)
            # name = "<<{}>>".format(emotion)
            # a.add(name, best_embeddings)
            # print(a.get_vector(name))

            # Show top 6 most similar in graph
            show_top = 10
            most_similar = a.similar_by_vector(best_embeddings, topn=show_top)
            print("Most similar to {}: {}".format(emotion, most_similar))

            first = True
            for ms in most_similar:
                label = ms[0]
                vec = a.get_vector(label)

                #We mark the most similar word by the class of emoji
                if first:
                    first = False
                    # most_similar_labels.append("<{}>:{}".format(emotion,label))
                    # print("Added {}".format("{} {}".format(idx2emoji[emotion].split("|")[0],label)))

                    # Note: we are adding the vector twice, once for the emoji and another for its closest word
                    emojis.append(idx2emoji[emotion].split("|")[0])
                    # emoji_vecs.append(best_embeddings)
                    emoji_vecs.append(vec)

                # else:
                most_similar_labels.append(label)
                most_similar_vec.append(vec)

            # most_similar_labels.append(idx2emoji[emotion].split("|")[0])
            # most_similar_labels.append("<<{}>>".format(emotion))
            # most_similar_vec.append(best_embeddings)

        most_similar_vec = most_similar_vec + emoji_vecs
        most_similar_labels = most_similar_labels + emojis


        # x,y = tsne(most_similar_vec)

        # classes = len(emojis)
        # x = x[:classes]
        # y = y[:classes]

        # emoji_xs = x[classes:]
        # emoji_ys = y[classes:]
        # emojis = most_similar_labels[classes:]

        # TensorFlow embedding projector already does the PCA/TSNE computation
        print("Writing projections")
        tf_projector_plot('most_similar_vec.tsv', 'most_similar_labels.tsv', most_similar_vec, most_similar_labels)

        print("words + emojis")

        limit = 500
        biggest_and_emoji_vec = tokens[:limit] + emoji_vecs
        biggest_and_emoji_labels = labels[:limit]+emojis
        x,y = tsne(biggest_and_emoji_vec)
        biggest_x, biggest_y , biggest_labels = x[:limit],y[:limit], biggest_and_emoji_labels[:limit]
        emoji_x, emoji_y, emoji_labels = x[limit:], y[limit:], biggest_and_emoji_labels[:limit]

        print(emojis)
        # exit(0)
        tsne_plot_seaborn_svg('most_similar', "Most similar", biggest_x,biggest_y, biggest_labels,emoji_x, emoji_y, emojis)

        # tsne_plot_seaborn_svg('most_similar', "Most similar", x,y, most_similar_labels,norms, emoji_xs, emoji_ys, emojis)
        # exit(0)

        # tsne_plot_seaborn(x,y,most_similar_labels)


        # plot only imagined vectors
        imagined_vecs =[]
        imagined_labels =[]
        for i,emotion in enumerate(idx2emoji.keys()):
            best_embeddings, _ = model.reverse(i)
            imagined_vecs.append(best_embeddings)
            imagined_labels.append(emotion)

        x, y = tsne(imagined_vecs)
        tsne_plot_seaborn(x,y, imagined_labels)


    if limit:
        tokens = tokens[:limit]
        labels = labels[:limit]

    # tsne_plot(tokens,labels)
    x, y = tsne(tokens)
    tsne_plot_seaborn(x,y,labels)

def tsne(tokens):
    tsne_model = TSNE(perplexity=60, n_components=2, init='pca', n_iter=5500, random_state=23)

    tokens = np.vstack(tokens)

    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    return x,y

def tf_projector_plot(vectors_file, labels_file , most_similar_vec, most_similar_labels):
    most_similar_vec = np.vstack(most_similar_vec)
    np.savetxt(vectors_file, most_similar_vec, delimiter="\t")
    # labels_encoded = [label for label in most_similar_labels]
    labels_encoded = [label.encode('utf-8') for label in most_similar_labels]
    np.savetxt(labels_file, labels_encoded, fmt='%s', delimiter="\t")

def tsne_plot_seaborn(x,y, labels):
    "Creates and TSNE model and plots it"

    plt.figure(figsize=(16, 16))
    df = pd.DataFrame({'x': x, 'y':y, 'group':labels})

    p=sns.scatterplot(x='x',y='y', data = df)#,fit_reg = False)
    # add annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p.annotate(labels[line],xy=(df.x[line], df.y[line]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',fontname='Apple Color Emoji')
        # p.text(df.x[line], df.y[line]+2, df.group[line], horizontalalignment='center', size='medium', color='black')

    plt.show()
# e_ - emoji...
def tsne_plot_seaborn_svg(name, title, x,y, labels, e_x, e_y, e_labels):
    # norms = [np.linalg.norm(token) for token in tokens]
    print(e_labels)

    write_geojson(name, x,y,labels,  e_x, e_y, e_labels)

    output_file("{}.html".format(name.replace("-","_")), title="{}".format(title))
    # output_file("{}.svg".format(name.replace("-","_")), title="{}".format(title))

    source = ColumnDataSource(data=dict(height=x,
                                        weight=y,
                                        names=labels))

    p = figure(title=title)#, x_range=Range1d(140, 275))
    p.scatter(x='weight', y='height', size=8, source=source)

    labels = LabelSet(x='weight', y='height', text='names', level='glyph',
                      x_offset=5, y_offset=5, source=source, render_mode='canvas')

    p.add_layout(labels)
    p.sizing_mode = 'scale_width'
    # p.output_backend = "svg"
    # p.output_backend = "svg"
    # export_svgs(p, filename="plot.svg")
    # p.add_layout(citation)

    show(p)
    print("Showed {}.html".format(name))
# cannot show emojsi
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

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def write_geojson(name,xs,ys,labels,e_x, e_y, e_labels):
    all_x = np.asarray(xs+e_x)
    all_y = np.asarray(ys+e_y)

    all_x *= 100.0 / all_x.max()
    all_y *= 100.0 / all_y.max()

    n_x = len(xs)
    n_y = len(ys)

    xs,ys = all_x[:n_x], all_y[:n_y]
    e_x,e_y = all_x[n_x:],all_y[n_y:]

    '''Emoji layer'''
    features = [create_feature(x, y, l) for (x, y, l) in list(zip(e_x, e_y, e_labels))]
    d = {
        "type": "FeatureCollection",
        "name": "test-points-short-named",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features}

    with open('{}_chunk_emoji.json'.format(name), 'w') as fp:
        json.dump(d, fp, cls=MyEncoder)
    '''end'''

    zoom_levels = 8
    partition_size = int(len(xs)/zoom_levels)

    for chunk_index, chunk in enumerate(chunks(list(zip(xs,ys,labels)), partition_size)):
        chunk_x, chunk_y, chunk_labels = zip(*chunk)

        features = [ create_feature(x,y,l) for (x,y,l) in list(zip(chunk_x,chunk_y,chunk_labels))]
        d = {
            "type": "FeatureCollection",
            "name": "test-points-short-named",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features}

        with open('{}_chunk_{}.json'.format(name,chunk_index), 'w') as fp:
            json.dump(d, fp, cls=MyEncoder)

def create_feature(x,y,label):
    return { "type": "Feature", "properties": { "name": "{}".format(label) }, "geometry": { "type": "Point", "coordinates": [x, y] } }

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

if __name__ == '__main__':
    config, models_config, preprocessing_config, test_config, args = parse_config()
    main(config,models_config,preprocessing_config, args)