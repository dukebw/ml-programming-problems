# Copyright 2018 Brendan Duke.
#
# This file is part of ML Programming Problems.
#
# ML Programming Problems is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# ML Programming Problems is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# ML Programming Problems. If not, see <http://www.gnu.org/licenses/>.

from glob import glob
import os

from nltk.tokenize import TweetTokenizer
import numpy as np


def test_naive_bayes():
    if not os.path.exists('YouTube-Spam-Collection-v1'):
        os.mkdir('YouTube-Spam-Collection-v1')
        os.system('wget http://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip')
        os.system('unzip YouTube-Spam-Collection-v1.zip -d ./YouTube-Spam-Collection-v1')
        os.system('rm -rf ./YouTube-Spam-Collection-v1/__MACOSX YouTube-Spam-Collection-v1.zip')

    tokenizer = TweetTokenizer(preserve_case=False,
                               reduce_len=True,
                               strip_handles=True)

    data_filepaths = glob('YouTube-Spam-Collection-v1/*')
    dataset_shards = []
    for path in data_filepaths:
        with open(path, 'r') as f:
            raw = f.read()

        shard = []
        for line in raw.strip().split('\n')[1:]:
            line = line.split(',')
            gt = int(line[-1])
            assert gt in [0, 1]

            features = tokenizer.tokenize(line[-2])
            shard.append((features, gt))

        dataset_shards.append(shard)

    train_set = []
    for shard in dataset_shards[:-1]:
        train_set += shard

    dev_set = dataset_shards[-1]

    word_counts = dict()
    for ex in train_set:
        for word in ex[0]:
            count = word_counts.get(word, 0)
            word_counts[word] = count + 1

    words = [w for w, count in word_counts.items() if count >= 3]
    word_to_index = {w: i for i, w in enumerate(words)}

    num_class = 2
    class_tally = np.zeros(num_class, dtype=np.int32)
    feat_class_tally = np.zeros([len(words), num_class], dtype=np.int32)

    for ex in train_set:
        gt = ex[1]
        for word in np.unique(ex[0]):
            if word not in word_to_index:
                continue

            feat_class_tally[word_to_index[word], gt] += 1

        class_tally[gt] += 1

    total_ex = np.sum(class_tally)
    w_theta = feat_class_tally.astype(np.float32) / total_ex
    w_pi = class_tally.astype(np.float32) / total_ex


if __name__ == '__main__':
    test_naive_bayes()
