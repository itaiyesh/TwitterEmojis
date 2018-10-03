import numpy as np
import torch
from utils import *

# comparing sum of positive labels to rest
def youtube_acc2(y_input, y_target):
    y_target = np.squeeze(y_target)

    happy_keys = ['smiling',
                  'heart',
                  'laughing',
                  'winking',
                  'angel',
                  'blushing',
                  'liplicking',
                  'relieved',
                  'inlove',
                  'kissing',
                  'playful_tongue_out',
                  'sun_glasses',
                  'excited_hands_shaking',
                  # 'shock_face',
                  # 'hungry_face',
                  'stary_eyes']

    ind = {k: i for i, k in enumerate(idx2emoji.keys())}
    happy_indexes = [ind[emotion] for emotion in happy_keys]

    predicted = [1 if 2 * label[happy_indexes].sum() > label.sum() else 0 for label in y_input]

    predicted = torch.from_numpy(np.array(predicted)).long().cuda()
    return float((predicted == y_target).sum()) / len(y_target)


def youtube_acc(y_input, y_target):
    y_target = np.squeeze(y_target)

    happy_keys = ['smiling',
                  'laughing',
                  'winking',
                  'angel',
                  'blushing',
                  'liplicking',
                  'calm_smiling',
                  'inlove',
                  'kissing',
                  'playful_tongue_out',
                  'sun_glasses',
                  'excited_hands_shaking',
                  # 'shock_face',
                  'hungry_face',
                  'stary_eyes']

    ind = {k: i for i, k in enumerate(idx2emoji.keys())}
    happy = [ind[emotion] for emotion in happy_keys]

    predicted = [np.argmax(label) for label in y_input]
    predicted = [1 if label in happy else 0 for label in predicted]

    # predicted = [1 if label[0] > label[17] else 0 for label in y_input]

    predicted = torch.from_numpy(np.array(predicted)).long().cuda()
    return float((predicted == y_target).sum()) / len(y_target)


def se0714_acc(y_input, y_target):
    # fear joy sadness

    y_target = np.squeeze(y_target)
    fear_keys = [
        'shock_face',
        'pale_face',
        'devil_face',
        'sweat_drop_face',
        'sick_face'
    ]

    joy_keys = ['smiling',
                'laughing',
                'winking',
                'angel',
                'blushing',
                'liplicking',
                'relieved',
                'inlove',
                'heart',
                'kissing',
                'playful_tongue_out',
                'sun_glasses',
                'excited_hands_shaking',
                # 'hungry_face',
                'stary_eyes']

    sadness_keys = [
        'sad_face',
        'weary',
        'crying_face',
        'mad_face',
        'exhausted']

    ind = {k: i for i, k in enumerate(idx2emoji.keys())}

    joy = [ind[emotion] for emotion in joy_keys]
    fear = [ind[emotion] for emotion in fear_keys]
    sad = [ind[emotion] for emotion in sadness_keys]

    partitions = [joy, fear, sad]

    # Minimum average probability required to belong to a class.
    threshold = 0.5
    predicted = [[1 if np.average(label[partition]) > threshold else 0 for partition in partitions] for label in
                 y_input]

    predicted = torch.from_numpy(np.array(predicted)).long().cuda()
    return float((predicted == y_target).sum()) / len(y_target)
