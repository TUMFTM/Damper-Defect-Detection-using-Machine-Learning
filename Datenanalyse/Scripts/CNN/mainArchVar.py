import argparse
import os
import sys
from datetime import datetime
import numpy as np
import time
from datetime import timedelta
from sklearn.model_selection import KFold
import joblib
sys.path.insert(0, os.path.abspath(os.path.join('../')))    # required to add path for importing from another directory
import utilities.preprocessing as prep
import utilities.plot as plotlib
import utilities.data_handling as datahandler
import CNN_framework
import utilities.evaluateCNN as evaluateCNN

#
#choices
models = ['baseline', 'baseline_new', 'doubleconv', 'sepConv', 'grouped_sepConv',
          'SqueezeNet', 'LeNet', 'TICNN', 'SE_ResNeXt', 'DamNet_v1', 'paper_prep', 'mlp']
prep1d = ['None', 'detrend', 'standard', 'zero2one', 'neg_one2one', 'fourier_channels', 'fourier_samples']
prep2d = ['greyscale', 'stft', 'recurrence_plot', 'gaf', 'wpi']
# argument parser
parser = argparse.ArgumentParser(description="script to run a single CNN model, specified in the arguments")
# data parameters
parser.add_argument("-logp", "--logspath", type=str, default='../../CNN/results/'+datetime.now().strftime("%Y_%m_%d_%Hh_%M")+'/',
                    help="path for the logged data (avoid absolute path if possible)")
parser.add_argument("--datapath", type=str, default='../../Datensatz/DataForCNN/data/DD2_raw_512_FlexRay_woCorruptedData/',
                    help="path to the dataset (include '/' at the end)")
parser.add_argument("--test_split", type=float, default=0.2,
                    help="percentage of dataset used as held out test data")
parser.add_argument("--val_split", type=float, default=0.2,
                    help="percentage of dataset used as held out test data")
parser.add_argument("-prep", "--preprocessing", type=str, default='fourier_channels',
                    choices=prep1d + prep2d, help="type of preprocessing to apply on data")
parser.add_argument("--sel_sensors", type=list,
                    default=['SPEED_FL', 'SPEED_FR', 'SPEED_RL', 'SPEED_RR', 'ACC_X', 'ACC_Y', 'YAW_RATE'],
                    help='list of strings with the sensor names to use (see Info.txt)')
parser.add_argument("-gpu", "--gpu_no", type=int, default=0, choices=[0, 1, 2],
                    help="GPU device number to train on")
parser.add_argument("-e", "--evaluate", action="store_true",
                    help="activate evaluation mode --> use trained net to classify a given data set")
parser.add_argument("-test", "--test_set", type=str, default='../../Datensatz/DataForCNN/data/testsets/DD_mass_raw_512_FlexRay_THP/',
                    help="path and file name of test set to evaluate")
parser.add_argument("--eval_modelpath", type=str,
                    default="//nas.ads.mwn.de/tumw/ftm/Projekte/Zehelein_Studenten/MA_Hemmert/Ergebnisse/robustness/dd2-c-256_damnet_1d_09ce870/")
parser.add_argument("-plain", "--plain_run", action="store_true",
                    help="run a plain_run instead of a crossvalidation (default is crossvalidation)")
# logging
parser.add_argument("-w", "--write_file", action="store_true",
                    help='redirects the print commands to log.txt in logspath')
parser.add_argument("-p", "--plot", action="store_true",
                    help="plots histograms and more")
parser.add_argument('-ex', "--export", action="store_true",
                    help="export the plots to tikz files")
parser.add_argument("-disp", "--display_steps", type=int, default=1,
                    help="how many training steps per epoch should be displayed (and logged!!)")
parser.add_argument("-meta", "--metadata", action="store_true",
                    help="enables full trace of a run including metadata such as memory and timing info during training")
# model setup
parser.add_argument("-in", "--input", type=str, default='1D', choices=['1D', '2D'],
                    help="control the input type")
parser.add_argument("-model", "--model", type=str, default='baseline',
                    choices=models,
                    help="name of the inference model to be used ")
# model hyperparameters
parser.add_argument("-epochs", "--num_epochs", type=int, default=650,
                    help='number of epochs, 1 epoch includes all training samples')
parser.add_argument("-bs", "--batch_size", type=int, default=128, choices=[64, 128, 256, 512],
                    help="number of training samples per mini-batch")
parser.add_argument("-lr", type=float, default=0.0003,
                    help="learning rate used for training (common choices between 0.1 and 0.0001)")
parser.add_argument("--enable_lr_decay", action="store_true",
                    help="Trigger decaying learning rate")
parser.add_argument("-l2", "--l2_str", type=float, default=0.00003,
                    help="L2 regularization strength")
parser.add_argument("--bias_init", type=float, default=0.01,
                    help="Initial value for constant initialization of bias (should be small (< 1))")
parser.add_argument("-k", "--keep_prob", type=float, default=0.5,
                    help="Probability to keep a single neuron in a dropout layer")
parser.add_argument("--es_patience", type=int, default=50,
                    help="number of epochs to wait for the validation error to decrease by es_mindelta")
parser.add_argument("--es_mindelta", type=float, default=0.001,
                    help="minimum required change to count delta in validation error as significant")
config = parser.parse_args()

config.plot = True
config.write_file = True
config.export = True
config.model = 'baseline_new'
config.depth = 1
config.kernels = 16

config.add_conv = 2
config.firstlayer_e = 1
config.pooling_size = 2

config.preprocessing = 'fourier_channels'
config.input = '1D'

config.sel_sensors = ['SPEED_FL', 'SPEED_FR', 'SPEED_RL', 'SPEED_RR', 'ACC_X', 'ACC_Y', 'YAW_RATE']

config.logspath = '../../BMW/CNN/results/RFE_FFT/Baseline/'+datetime.now().strftime("%Y_%m_%d_%Hh_%M")+'/'

# config.preprocessing = 'stft'
# config.input = '2D'

### Input for model evaluation ---- start
# config.evaluate = True
# config.eval_modelpath = '../../CNN/results/2020_04_01_23h_15_FFT/'
config.eval_modelpath = '../../CNN/results/2020_04_01_21h_15_Detrend/'
config.test_set = '../../Datensatz/DataForCNN/data/testsets/DD_mass_raw_512_FlexRay_TZ/'
# config.test_set = '../../Datensatz/DataForCNN/data/testsets/DD_ReiterEngineering_raw_512_FlexRay/'
### Input for model evaluation ---- end

# Results from preprocessing optimization
l2_prepAnalysis = {'detrend': 0.003324749879539013,
                   'fourier_channels': 0.00025678626843728125,
                   'fourier_samples': 0.008149031549692154,
                   'gaussian': 0.01,
                   'neg_one2one': 0.0002422870536211569,
                   'None': 0.0010284192488334493,
                   'standard': 0.0032478600402736624,
                   'greyscale': 0.00028467760421335697,
                   'recurrence_plot': 0.00003,
                   'stft': 1.2270923434698489e-05,
                   'wpi': 0.00015223154332488775
                   }

lr_prepAnalysis = {'detrend': 0.0019805091433227062,
                   'fourier_channels': 0.0012941349996253848,
                   'fourier_samples': 2.0400091671035625e-05,
                   'gaussian': 0.0008412579360225878,
                   'neg_one2one': 0.0005970468505575491,
                   'None': 0.0011377801222186023,
                   'standard': 0.00040961230150153317,
                   'greyscale': 0.005820241756737232,
                   'recurrence_plot': 0.0003,
                   'stft': 5.971175050945021e-05,
                   'wpi': 0.00019125331891700625
                   }

config.l2_str = l2_prepAnalysis[config.preprocessing]
config.lr = lr_prepAnalysis[config.preprocessing]

config.featureviz = 'CNN_Model/conv1_relu_bn/conv1D/BiasAdd:0'  # operation name of the layer to be visualized
config.metrics_avg = 'weighted'
# +++++ end of block +++++

# ensure proper configuration
if config.input == '1D' and config.preprocessing not in prep1d:
    raise ValueError("specified preprocessing not available for 1D inputs")
if config.input == '2D' and config.preprocessing not in prep2d:
    raise ValueError("specified preprocessing not available for 2D inputs")

# train on different GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_no)


def main(argv=None):  # pylint: disable=unused-argument
    if config.evaluate:

        evaluateCNN.evaluateCNN(config=config)

    else:
        if not os.path.isdir(config.logspath) and (config.plot or config.write_file):
            os.makedirs(config.logspath)  # make sure directory exists for writing the file

        ### do the data handling
        config.orig_seq_lngth, _, config.avail_sensors = datahandler.read_infotxt(config.datapath)
        # get the data and info
        start_time = time.time()

        alltrain, test_set, labels, config.num_ch, config.seq_lngth, config.sensors, config.img_dim, scaler = \
                datahandler.handle_data(config.datapath, config.sel_sensors,
                                        file_format='csv', preprocessing_type=config.preprocessing, split_ratio=config.test_split)

        # save scaler
        scaler_filename = config.logspath + 'scaler.save'
        joblib.dump(scaler, scaler_filename)

        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage for data handling: " + str(timedelta(seconds=int(round(time_dif)))))
        config.num_cl = len(labels)  # obtain number of classes
        if config.plot:
            # plot different preprocessing types (single one per prep. type)
            if not os.path.isdir(config.logspath):
                os.makedirs(config.logspath)  # make sure directory exists for writing the log file
            if config.input == '2D':
                plotlib.plot_image_sample(alltrain, labels, config.sensors, 2, config.img_dim, config.logspath,
                                          fileext='')
            else:
                plotlib.plot_sample(alltrain, 2, labels, config.sensors, config.seq_lngth, config.logspath,
                                    filenameext='')
        ### data handling done

        ### run
        # base = config.logspath
        # for config.add_conv in [2, 1, 0]:
        #     for config.firstlayer_e in [1, 2, 3, 7, 15, 31, 63]:      # [2, 3, 7, 15, 31, 63]
        #         for config.pooling_size in [1, 2, 4, 8, 16, 24, 32]:        # [1, 2, 4, 8, 16, 24, 32]
        #             config.logspath = base + 'addc_' + str(config.add_conv) + '_e_' + str(config.firstlayer_e).zfill(3) + '_poolingsize_' + str(config.pooling_size).zfill(3) + '/'
        #             if config.plain_run:
        #                 result = plain_run(alltrain, test_set, labels)
        #             else:
        #                 result = crossvalidate_run(alltrain, test_set, labels, splits=5)
        #             config.logspath = base



        base = config.logspath
        config.logspath = base + 'addc_' + str(config.add_conv) + '_e_' + str(config.firstlayer_e).zfill(3) + '_poolingsize_' + str(config.pooling_size).zfill(3) + '/'
        if config.plain_run:
            result = plain_run(alltrain, test_set, labels)
        else:
            result = crossvalidate_run(alltrain, test_set, labels, splits=5)
        config.logspath = base + '/'
        # result = datasetsize_inspectionrun(alltrain, test_set, labels, splits=5)
        print("Final test accuracy: %s" % result)


def plain_run(alltrain, test_set, labels):
    testaccuracy, trainaccuracy, trainaccuracy_last, test_cost, train_cost, times = [], [], [], [], [], []  # for saving results and to generate a CV_log.txt file
    # split whole "train" set into train and validation
    train, val = datahandler.split_data(alltrain.data, alltrain.labels, config.val_split)
    config.disp_train = int(train.num_examples / (config.display_steps * config.batch_size))  # steps per epoch regardless batch size
    dataset = datahandler.make_datasets(train, val, test_set)

    testacc, trainacc, trainacc_last, duration, valacc =  \
        CNN_framework.run(config,
                          dataset=dataset,
                          classlabels=labels)
    # plot histograms
    if config.plot:
        plotlib.plot_histogram(dataset.test.labels, 'Test data', config.logspath, 'test_histogram', labels, export=config.export)
        plotlib.plot_histogram(dataset.train.labels, 'Training data', config.logspath, 'train_histogram', labels, export=config.export)
        plotlib.plot_histogram(dataset.validation.labels, 'Validation data', config.logspath, 'validation_histogram', labels, export=config.export)
    print("test accuracy: %s" % testacc)
    print("train accuracy: %s" % trainacc)
    print("last train acc: %s" % trainacc_last)

    if config.write_file:
        cvlog = open(config.logspath + 'CV_log.txt', 'w+')
        cvlog.write("test accuracies: %s\n" % testacc)
        cvlog.write("train accuracies: %s\n" % trainacc)
        cvlog.write("last train acc: %s\n" % trainacc_last)
        cvlog.write("average test accuracy: %s" % testacc)
        cvlog.write("average training time: %s" % duration)
        cvlog.close()

    return testacc


def crossvalidate_run(train_dataset, test_set, labels, splits=10):
    testaccuracy, trainaccuracy, trainaccuracy_last, test_cost, train_cost, times = [], [], [], [], [], []  # for saving results of different runs
    """function to use 'splits'-fold cross validation, writing results to CV_log.txt"""
    basepath = config.logspath[:-1] + "_cv" + str(splits) + '/'
    fold = 0
    kf = KFold(n_splits=splits, shuffle=True)  # initialize k-fold cross validation

    if config.write_file or config.plot:
        if not os.path.isdir(basepath):
            os.makedirs(basepath)  # make sure directory exists for writing the log file
        cvlog = open(basepath + '/CV_log.txt', 'w+')

    # get the data
    if config.plot:  # test set stays same, but train/val is different for every split
        plotlib.plot_histogram(test_set.labels, 'Test data', basepath, 'test_histogram', labels)

    # cross validation loop
    for train_idx, val_idx in kf.split(train_dataset.data, train_dataset.labels):
        # construct data set with train/val data for this "fold"
        train_set = datahandler.make_dataset(train_dataset.data[train_idx], train_dataset.labels[train_idx], train_dataset.index[train_idx])
        val_set = datahandler.make_dataset(train_dataset.data[val_idx], train_dataset.labels[val_idx], train_dataset.index[val_idx])
        # set own path for each run
        config.logspath = basepath + 'fold_' + str(fold) + '/'
        config.disp_train = int(train_set.num_examples / (config.display_steps * config.batch_size))  # steps per epoch regardless batch size
        # run optimization and evaluation procedure
        testacc, trainacc, trainacc_last, duration, valacc = \
            CNN_framework.run(config,
                              dataset=datahandler.make_datasets(train_set, val_set, test_set),
                              classlabels=labels)
        # plot histograms
        if config.plot:
            plotlib.plot_histogram(train_set.labels, 'Training data', config.logspath, 'train_histogram',
                                   labels)
            plotlib.plot_histogram(val_set.labels, 'Validation data', config.logspath,
                                   'validation_histogram',
                                   labels)
        testaccuracy.append(testacc)
        trainaccuracy.append(trainacc)
        trainaccuracy_last.append(trainacc_last)
        times.append(duration)
        print("test accuracy in fold %s: %s " % (fold, testacc))
        print("train accuracy in fold %s: %s" % (fold, trainacc))
        print("last train acc. in fold %s: %s" % (fold, trainacc_last))
        fold += 1
    mean_testacc = np.mean(testaccuracy)
    if config.write_file:
        cvlog.write("test accuracies: %s\n" % testaccuracy)
        cvlog.write("train accuracies: %s\n" % trainaccuracy)
        cvlog.write("last train acc: %s\n" % trainaccuracy_last)
        cvlog.write("average test accuracy: %s" % mean_testacc)
        cvlog.write("average training time: %s" % np.mean(times))
        cvlog.close()
    return mean_testacc


def datasetsize_inspectionrun(alltrain, test_set, labels, splits=5):
    orig_path = config.logspath
    # scale such that remaining _train_ set holds power of 2 data samples
    sizes = (np.ceil(np.asarray([256, 512, 1024, 2048, 4096, 8192, 10000]) * splits / (splits - 1)))
    sizes = [int(i) for i in sizes]     # make them integers
    i = 0
    for size in sizes:
        config.logspath = orig_path + 'tr_size' + str(sizes[i]) + '/'
        traindata_trunc = alltrain.data[:size]
        trainlabels_trunc = alltrain.labels[:size]
        trainindices_trunc = alltrain.index[:size]
        result = crossvalidate_run(datahandler.make_dataset(traindata_trunc, trainlabels_trunc, trainindices_trunc), test_set, labels, splits)
        i += 1
    return result


if __name__ == '__main__':
    main(config)
