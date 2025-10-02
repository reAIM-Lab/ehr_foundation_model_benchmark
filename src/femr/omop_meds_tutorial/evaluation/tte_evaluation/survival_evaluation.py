from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import torchtuples as tt

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

base = '/user/zj2398/cache/mtpp_8k/results/'
# phenotype_list = ['masld']
phenotype_list = ['stroke']
model = 'deephit_no_divide_mask_mean_all'

# base = '/data/processed_datasets/processed_datasets/mimic/phenotype/'
# phenotype = 'masld'
# model = 'motor'

# path = base + phenotype + '/' + model + '/features_with_label/'

# Use 10% for validation
for phenotype in phenotype_list:
    path = base + phenotype + '/' + model + '/features_with_label/'
    train = pd.read_parquet(path + 'train.parquet')
    val = train.sample(frac = 0.1, replace = False)
    train = train.drop(val.index)
    test = pd.read_parquet(path + 'test.parquet')
    # print(train)
    train['time_to_event_days'].max()/365
    train[~train.boolean_value].time_to_event_days.plot.hist(bins = 1000)

    # Transform labels
    num_durations = 100
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['time_to_event_days'].values, df['censor_label'].values) 

    # Extract features and labels
    x_train = np.stack(train.features.values)
    y_train = labtrans.fit_transform(*get_target(train))
    train = (x_train, y_train)

    x_val = np.stack(val.features.values)
    y_val = labtrans.fit_transform(*get_target(val))
    val = (x_val, y_val)

    x_test = np.stack(test.features.values)
    durations_test, events_test = get_target(test)

    # Define one layer NN for Deephit
    in_features = x_train.shape[1]
    out_features = labtrans.out_features

    net = tt.practical.MLPVanilla(in_features, [], out_features)

    # Train Deephit
    model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
    epochs, batch = 100, 2048
    log = model.fit(x_train, y_train, batch, epochs, [tt.callbacks.EarlyStopping()], val_data=val)

    # Predict on test set
    surv = model.predict_surv_df(x_test)

    # Compute boostrapped performance
    results = {'C-Index':[], 'Int-Brier':[]}
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    for boot in tqdm(range(100)):
        sample = np.random.choice(len(x_test), len(x_test))
        ev = EvalSurv(surv[sample], durations_test[sample], events_test[sample], censor_surv='km')
        results['C-Index'].append(ev.concordance_td('antolini'))
        results['Int-Brier'].append(ev.integrated_brier_score(time_grid))

    # Display
    for metric in results:
        print('{}: {:.2f} ({:.2f})'.format(metric, np.mean(results[metric]), np.std(results[metric])))