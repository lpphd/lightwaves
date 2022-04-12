import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV

import glob
from timeit import default_timer as timer

from sklearn.metrics import f1_score, accuracy_score
from sktime.datasets import load_from_arff_to_dataframe
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested, from_nested_to_3d_numpy
from sktime.transformations.panel.rocket import MiniRocketMultivariate

dir_prefix = "."

stats = []
for dir in sorted(glob.glob(F"{dir_prefix}/ARFFDatasets/*/")):
    dataset = dir.split("/")[-2]
    if dataset in ["InsectWingbeat"]:
        continue
    train_path = dir + dataset + "_TRAIN.arff"
    test_path = dir + dataset + "_TEST.arff"
    train_x, train_y = load_from_arff_to_dataframe(train_path)
    test_x, test_y = load_from_arff_to_dataframe(test_path)

    tslen = len(train_x.loc[0]['dim_0'])
    if tslen < 9:
        train_x = from_nested_to_3d_numpy(train_x)
        test_x = from_nested_to_3d_numpy(test_x)
        train_x = np.pad(train_x, ((0, 0), (0, 0), (0, 9 - tslen)))
        test_x = np.pad(test_x, ((0, 0), (0, 0), (0, 9 - tslen)))
        train_x = from_3d_numpy_to_nested(train_x)
        test_x = from_3d_numpy_to_nested(test_x)

    for seed in range(30):
        print(F"Dataset : {dataset} - Seed {seed}", flush=True)
        np.random.seed(seed)

        rocket = MiniRocketMultivariate(random_state=seed)

        start = timer()
        rocket.fit(train_x)
        end = timer()

        parameter_generation_time = end - start

        start = timer()
        X_training_transform = np.nan_to_num(rocket.transform(train_x), posinf=0, neginf=0)
        end = timer()

        kernel_application_time = end - start

        start = timer()

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(X_training_transform, train_y)

        end = timer()

        training_time = end - start

        start = timer()
        X_test_transform = np.nan_to_num(rocket.transform(test_x), posinf=0, neginf=0)
        predictions = classifier.predict(X_test_transform)
        end = timer()
        inference_time = end - start

        acc = accuracy_score(test_y, predictions)
        wf1 = f1_score(test_y, predictions, average='weighted')

        stats.append(
            [dataset, seed, parameter_generation_time, kernel_application_time, training_time, inference_time, acc,
             wf1])
        stats_df = pd.DataFrame.from_records(stats, columns=['Dataset', 'Seed', 'Parameter Generation Time',
                                                             'Train set transformation time', 'Training time',
                                                             'Inference time', 'Accuracy', 'Weighted F1'])
        stats_df.to_csv("minirocket_uea_metrics.csv", index=False)
