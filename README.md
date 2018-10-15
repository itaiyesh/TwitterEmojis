# Predicting Text Sentiment Using Tweets Embedded Emojis

A project in Artificial Intelligence - 236502

# Requirements
- pyTorch 4.0 installed with CUDA
- h5py (https://www.h5py.org)
- pandas
- pymongo // only needed for the data gathering phase
- gensim
- sklearn
- tqdm
- matplotlib
- bokeh
- seaborn

Due to licensing restrictions, we cannot release the data sets used for training. 
However, it is possible to test the trained model on the various public data sets provided here.

# Running the experiments
- To run all experiment with the trained model, use
    ```sh
        $ test_config config/test_config.json --resume "saved\lstm\model_best.pth.tar"
    ```

# Project structure
- The project template is derived from [this](https://github.com/victoresque/pytorch-template) repo.
- base - contains files base classes for data loader, model and trainer
- data_loader - includes data loaders for both the training and the testing phases.
- datasets
    - raw - includes the raw datasets of tweets in an HDF5 format.
    - processed 
        - bow.hdf5 - includes the processed tweets in a bag-of-word format (to be used by the bag-of-words models)
        - sequence.hdf5 - includes the processed tweets in a sequence format (to be used by the biLSTM.)
        - labels.hdf5 - contains labels for both bow.hdf5 and bow.hdf5 files.
        - vocab.hdf5 - contains the vocab of 24k words computed for the training set.
    - tests - contain the various known data sets used in the experiment (Note: we did not release these data sets)
        - test_datasets.py - wrapper around the various data sets found in ../tests folder
        - tweets_dataset.py - wrapper around tweets dataset used for training.
- config
    - lstm.json, nb.json, svme.json, svmew2v.json - config files for training the model during the experiment
    - preprocessing_config.json - includes all parameters used for collecting and preprocessing the tweets.
    - models_config.json - includes parameters (i.e. embedding dimension) for all models.
    - test_config.json - defines the test data sets and metrics for testing the model.
- model 
    - model.py - contains all model implementations.
    - loss.py - contain loss functions used throughout the experiment.
    - metric.py - contain different implementations of different metrics (i.e. accuracy)
- saved - location where run log and model checkpoint from the experiment are output to.
- tester - includes the logic managing all testing experiments once the model has trained.
- trainer - contains the basic trainer implementation.
- utils -
    - index_to_emojiy.py - contain the mapping of index to emoji groups.
    - parsing_utils.py - implementation of all utility functions used to retrieve and query label information from tweets.
    - test_metric.py - metric used in test sets.
    - util.py - global operations such as parsing of user arguments etc...
    - vocabulary.py - implementation of Vocabulary class which is where most tokenization and text cleaning is taking place.
- prepare_data.py - all collecting and preprocessing logic is implemented here.
- train.py - main script used for training a model once the data is processed.
- test.py - main script used to run all experiments.
- visualize_data - main script used to generate graphs from training results.

