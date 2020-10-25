import numpy as np
import os
import itertools
import random
import math
import tensorflow as tf
import re
from tensorboard.backend.event_processing import event_accumulator
# mpl.use("pgf")
# pgf_norcfonts = {"pgf.rcfonts": False}  # use document font for pgfs
# mpl.rcParams.update(pgf_norcfonts)
import matplotlib.pyplot as plt
import utilities.preprocessing as prep
from matplotlib2tikz import save as tikz_save
plt.gray()  # set grayscale

###  define colors
# main
TUMmain = '#0065BD'
TUMwhite = '#ffffff'
TUMblack = '#000000'
# secondary
TUMgray1 = '#333333'
TUMgray2 = '#808080'
TUMgray3 = '#CCCCC6'
TUMblue1 = '#005293'
TUMblue2 = '#003359'
# accent
TUMaccent1 = '#64A0C8'
TUMaccent2 = '#98C6EA'
TUMaccent3 = '#DAD7CB'
TUMaccent4 = '#A2AD00'
TUMaccent5 = '#E37222'


def plot_histogram(data, title, folder, filename, target_names, export=False):
    """plot a histogram and save as pdf and pgf file
    :param data: the data
    :param title: title for the plot
    :param folder: path to the folder to save at (use '/' at the end)
    :param filename: filename(no file extension)
    :param target_names: list with labels for the ticks
    :param export: enable tikz export
    """
    plt.hist(np.argmax(data, axis=1), color=TUMgray2, bins=np.arange(len(target_names)+0.5), align='left', rwidth=0.8)
    # plt.title('Histogramm ' + title)
    plt.xlabel("Label")
    plt.ylabel("Häufigkeit")
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.savefig(folder + filename + '.pdf')
    if export:
        tikz_save(folder + filename + '.tikz', encoding='utf8')
    plt.close()


def plot_confusionmatrix(cm, folder, target_names, normalize=False, export=False, savefig=True, filenamePostfix='', closefig=True):
    """ plot a confusion matrix
    :param cm the confusion matrix
    :param folder path to storage folder
    :param target_names the class labels
    :param normalize bool to apply normalization or not
    :type folder str
    :type target_names list
    :type normalize bool
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greys)

    # Make various adjustments to the plot.
    plt.title('Konfusionsmatrix')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)  # , rotation=45)
    plt.yticks(tick_marks, target_names)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Ermittelte Klasse')
    plt.ylabel('Tatsächliche Klasse')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    if savefig:
        plt.savefig(folder + 'Confusionmatrix' + filenamePostfix + '.pdf')
    if export:
        tikz_save(folder + 'confmat' + filenamePostfix + '.tikz', encoding='utf8', show_info=False)
    if closefig:
        plt.close()


def plot_tensorboard_scalar(eventlogpath,
                            value,  # Error, Accuracy, Cost
                            export=False
                            ):
    """plotting a scalar from tensorboard
    :param eventlogpath path to event log file (include '/' at the end)
    :param value: specify the value to be printed: 'Error', 'Accuracy' or 'Cost'
    :param export: enable tikz export
    :type eventlogpath str
    :type value str
    :type export: bool
    """
    path_train = eventlogpath + '/graph/train/'
    path_val = eventlogpath + '/graph/val/'

    ea_train = event_accumulator.EventAccumulator(path_train)
    ea_val = event_accumulator.EventAccumulator(path_val)
    ea_train.Reload()
    ea_val.Reload()

    if value == 'Error':
        scalar = 'CNN_Model/Accuracy/Error'
    elif value == 'Accuracy':
        scalar = 'CNN_Model/Accuracy/Accuracy'
    elif value == 'Cost':
        scalar = 'CNN_Model/Cost'
    else:
        raise ValueError('specified value %s not defined', value)

    train_scalar = [(s.step, s.value) for s in ea_train.Scalars(scalar)]
    val_scalar = [(s.step, s.value) for s in ea_val.Scalars(scalar)]

    train_scalar_x = [x[0] for x in train_scalar]
    train_scalar_y = [x[1] for x in train_scalar]
    val_scalar_x = [x[0] for x in val_scalar]
    val_scalar_y = [x[1] for x in val_scalar]

    plt.plot(train_scalar_x, train_scalar_y, label='Training', color=TUMgray1, linestyle='--', linewidth=1.5)
    plt.plot(val_scalar_x, val_scalar_y, label='Validierung ', color=TUMblack, linestyle='-', linewidth=1.5)
    plt.xlabel('Iterationen')
    plt.ylabel('Klassifikationsfehler' if value == 'Error' else 'Kosten')
    plt.grid(True)
    plt.legend()
    plt.savefig(eventlogpath + '/plot_' + value + '.pdf')
    if export:
        tikz_save(eventlogpath + '/plot_' + value + '.tikz', encoding='utf8')
    plt.close()
    # plot_tensorboard_scalar('logs/1D_CNN/2018_07_27_14h_04','Error')


def plot_data(x, y, name, xlabel, ylabel, savepath):
    plt.plot(x, y, label=name, color=TUMmain)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(savepath + name + '.pdf')
    plt.close()


def plot_sample(dataset, sample, labels, sensors, seq_length, savepath, filenameext=None, export=False, show=False):
    """plot a random sample from the dataset
    :param dataset the dataset to plot from
    :type dataset Dataset
    :param labels one-hot encoded labels of the corresponding dataset
    :param sensors list containing the names of the sensors
    :type sensors list
    :param seq_length the length of a sequence of one single sensor
    :type seq_length int
    :param savepath path to save the images at
    :type savepath str
    :param export: enable tikz export
    :type export: bool
    """
    dataset.classidx = np.argmax(dataset.labels, axis=1)

    classification_label = labels[dataset.classidx[sample]]
    xlabel = 'Datenpunkte bei 50 Hz'
    sample_multichannel = np.reshape(dataset.data[sample], (len(sensors), seq_length))
    if all(x in sensors for x in ['SPEED_FL', 'SPEED_FR', 'SPEED_RL', 'SPEED_RR']):
        speed_fl_idx = sensors.index('SPEED_FL')
        speed_fr_idx = sensors.index('SPEED_FR')
        speed_rl_idx = sensors.index('SPEED_RL')
        speed_rr_idx = sensors.index('SPEED_RR')

        f, axarr = plt.subplots(2, 2, sharex='col', sharey='all')
        plt.suptitle("Printed sample %s, Label of data sample: %s" % (sample, str(classification_label)))
        axarr[0, 0].plot(sample_multichannel[speed_fl_idx], color=TUMblack)
        axarr[0, 0].set_title('$n_\mathrm{FL}$')
        axarr[0, 0].grid(True, which='major')
        axarr[0, 1].plot(sample_multichannel[speed_fr_idx], color=TUMblack)
        axarr[0, 1].set_title('$n_\mathrm{FR}$')
        axarr[0, 1].grid(True, which='major')
        axarr[1, 0].plot(sample_multichannel[speed_rl_idx], color=TUMblack)
        axarr[1, 0].set_title('$n_\mathrm{RL}$')
        axarr[1, 0].grid(True, which='major')
        axarr[1, 1].plot(sample_multichannel[speed_rr_idx], color=TUMblack)
        axarr[1, 1].set_title('$n_\mathrm{RR}$')
        axarr[1, 1].grid(True, which='major')
        for ax in axarr.flat:
            ax.set(xlabel=xlabel, ylabel='$n$ in rpm')
            ax.set_xlim([0, seq_length])
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axarr.flat:
            ax.label_outer()
        plt.savefig(savepath + '/sample_rpm_1d' + filenameext + '.pdf')
        if export:
            tikz_save(savepath + '/sample_rpm_1d' + filenameext + '.tikz', encoding='utf8', show_info=False)
        if show:
            plt.show()
        plt.close()
    if all(x in sensors for x in ['ACC_X', 'ACC_Y']):
        accel_x_idx = sensors.index('ACC_X')
        accel_y_idx = sensors.index('ACC_Y')

        f, axarr = plt.subplots(2, sharex='col')
        plt.suptitle("Label of data sample: " + str(classification_label))
        axarr[0].plot(sample_multichannel[accel_x_idx], color=TUMblack)
        axarr[0].set_title(sensors[accel_x_idx])
        axarr[1].plot(sample_multichannel[accel_y_idx], color=TUMblack)
        axarr[1].set_title(sensors[accel_y_idx])
        axarr[1].set_xlabel(xlabel=xlabel)

        for ax in axarr:
            ax.label_outer()
            ax.set_xlim([0, seq_length])
            ax.set(ylabel=r'Beschleunigung in $\frac{m}{s^2}$')
        plt.savefig(savepath + '/sample_acc_1d' + filenameext + '.pdf')
        if export:
            tikz_save(savepath + '/sample_acc_1d' + filenameext + '.tikz', encoding='utf8', show_info=False)
        plt.close()
    if 'YAW_RATE' in sensors:
        yaw_idx = sensors.index('YAW_RATE')
        plt.plot(sample_multichannel[yaw_idx], color=TUMblack)
        plt.title("Label of data sample: " + str(classification_label))
        plt.xlabel(xlabel)
        plt.xlim([0, seq_length])
        plt.ylabel(r'Gierrate in $\frac{rad}{s}$')
        plt.savefig(savepath + '/sample_yaw_1d' + filenameext + '.pdf')
        if export:
            tikz_save(savepath + '/sample_yaw_1d' + filenameext + '.tikz', encoding='utf8', show_info=False)
        plt.close()


def plot_conv_layer_1d(session, x, dataset, sample, savepath, layer, export=False):
    """plot the result of a convolution1d layer"""
    # values2 = session.run('CNN_Model/conv1_relu_bn/ReLU:0', feed_dict=feed_dict)

    # Calculate and retrieve the output values of the layer for given input sample
    feed_dict = {x: [dataset.data[sample]]}
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer. result has shape (?, seq_length, n_filters)
    num_filters = values.shape[2]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))

    # Create figure with a grid of sub-plots.
    f, axes = plt.subplots(num_grids, num_grids, sharex='col')
    plt.suptitle("Visualization of Results in " + str(layer))
    f.subplots_adjust(wspace=0.5)
    f.text(0.5, 0.04, 'Datenpunkte bei 50 Hz', va='center', ha='center')
    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            out = values[0, :, i]
            ax.set_xlim([0, values.shape[1]])
            # Plot image.
            ax.plot(out, linewidth=1.0)

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.savefig(savepath + '/sample_conv1_filters' + '.pdf')
    if export:
        tikz_save(savepath + '/sample_conv1_filters' + '.tikz', encoding='utf8', show_info=False)
    plt.close()


def plot_conv_layer_1d_spectrum(session, x, dataset, sample, savepath, layer, export=False):
    """plot the result of a convolution1d layer"""
    # values2 = session.run('CNN_Model/conv1_relu_bn/ReLU:0', feed_dict=feed_dict)

    # Calculate and retrieve the output values of the layer for given input sample
    feed_dict = {x: [dataset.data[sample]]}
    values = session.run(layer, feed_dict=feed_dict)
    fft_data, fft_length = prep.fourier(values[0].T, 1, values.shape[1])
    fft_data = fft_data.T
    # Number of filters used in the conv. layer. result has shape (?, seq_length, n_filters)
    num_filters = values.shape[2]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))

    # Create figure with a grid of sub-plots.
    f, axes = plt.subplots(num_grids, num_grids, sharex='col')
    plt.suptitle("Visualization of Results in " + str(layer))
    f.subplots_adjust(wspace=0.5)
    f.text(0.5, 0.04, 'Fourier Transformation der Feature Maps', va='center', ha='center')
    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            out = (fft_data[:, i])
            ax.set_xlim([0, fft_length])
            # Plot image.
            ax.plot(out, linewidth=1.0)

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.savefig(savepath + '/sample_conv1_filters_fft' + '.pdf')
    if export:
        tikz_save(savepath + '/sample_conv1_filters_fft' + '.tikz', encoding='utf8', show_info=False)
    plt.close()


def plot_conv_layer_1d_weights(session, layer, savepath, export=False):
    tensorlist = [v for v in tf.trainable_variables() if v.name.endswith('kernel:0') and layer in v.name]
    # there should only be one entry in the list, if multiple, take first
    width = int(tensorlist[0].shape[0])
    depth = int(tensorlist[0].shape[1])
    num = int(tensorlist[0].shape[2])
    weights = session.run(tensorlist[0].op.name + ':0')     # get the tensor values, note that no feed dict required here
    num_grids = int(math.ceil(math.sqrt(num)))   # for printing only sqrt(num) of the num filters

    f, axes = plt.subplots(num_grids, depth, sharex='col', sharey='row')
    f.subplots_adjust(wspace=0.5)
    plt.suptitle("Visualization of weights in " + str(layer))

    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        out = weights[:, i % depth, math.floor(i/depth)]    # print channels column-wise, take the first few filters
        ax.set_xlim([0, width-1])   # with is integer
        # Plot image.
        ax.plot(out, linewidth=1.0)

    plt.savefig(savepath + '/sample_conv1_filter_weights' + '.pdf')
    if export:
        tikz_save(savepath + '/sample_conv1_filter_weights' + '.tikz', encoding='utf8', show_info=False)
    plt.close()


def plot_image_sample(dataset, labels, sensors, sample, img_dim, savepath, fileext=None, show=False, export=False):
    """
    function to print a single stft-transformed sample, just prints the first channel
    :param dataset: the raw data
    :type dataset: Dataset
    :param labels: list containing all existing labels in the dataset
    :type labels: list
    :param sensors: list with the names of available sensors
    :type sensors: list
    :param sample: index of the sample to plot
    :type sample: int
    :param img_dim: the image dimensions
    :type img_dim: (int, int)
    :param savepath: path to save the image at
    :type savepath: str
    :return: nothing
    """
    sample_data = np.squeeze(dataset.data[sample])
    sample_label = labels[int(np.argmax(dataset.labels[sample]))]
    sample_multichannel = np.reshape(sample_data, (len(sensors), img_dim[0], img_dim[1]))
    if all(x in sensors for x in ['SPEED_FL', 'SPEED_FR', 'SPEED_RL', 'SPEED_RR']):
        speed_fl_idx = sensors.index('SPEED_FL')
        speed_fr_idx = sensors.index('SPEED_FR')
        speed_rl_idx = sensors.index('SPEED_RL')
        speed_rr_idx = sensors.index('SPEED_RR')

        f, axarr = plt.subplots(2, 2, sharex='col', sharey='row')
        f.subplots_adjust(wspace=-0.25)
        axarr[0, 0].imshow(sample_multichannel[speed_fl_idx])
        axarr[0, 0].set_title('$n_\mathrm{FL}$')
        axarr[0, 1].imshow(sample_multichannel[speed_fr_idx])
        axarr[0, 1].set_title('$n_\mathrm{FR}$')
        axarr[1, 0].imshow(sample_multichannel[speed_rl_idx])
        axarr[1, 0].set_title('$n_\mathrm{RL}$')
        axarr[1, 1].imshow(sample_multichannel[speed_rr_idx])
        axarr[1, 1].set_title('$n_\mathrm{RR}$')
        plt.suptitle('Printed sample %s, corresponding label: %s' % (sample, sample_label))
        # plt.tight_layout()
        plt.savefig(savepath + '/sample_speeds_2d' + '.pdf')
        if export:
            tikz_save(savepath + '/sample_speeds_2d' + fileext + '.tikz', encoding='utf8', show_info=False)
        if show:
            plt.show()
        plt.close()

    if all(x in sensors for x in ['ACC_X', 'ACC_Y']):
        accel_x_idx = sensors.index('ACC_X')
        accel_y_idx = sensors.index('ACC_Y')

        f, axarr = plt.subplots(2, sharex='col', sharey='row')
        plt.suptitle('Printed sample %s, corresponding label: %s' % (sample, sample_label))
        axarr[0].imshow(sample_multichannel[accel_x_idx])
        axarr[0].set_title(sensors[accel_x_idx])
        axarr[1].imshow(sample_multichannel[accel_y_idx])
        axarr[1].set_title(sensors[accel_y_idx])
        axarr[1].set_xlabel(xlabel="Zeit in Datenpunkten")
        plt.savefig(savepath + '/sample_acc_2d' + '.pdf')
        if show:
            plt.show()
        plt.close()

    if 'YAW_RATE' in sensors:
        yaw_idx = sensors.index('YAW_RATE')
        plt.title('Printed sample %s, corresponding label: %s' % (sample, sample_label))
        plt.imshow(sample_multichannel[yaw_idx])
        plt.savefig(savepath + '/sample_yaw_2d' + '.pdf')
        if show:
            plt.show()
        plt.close()


def plot_kernels(path='', tensor_name='', show_fig = False, save_fig = True, export2tikz = False):
    """
    function to plot all trained kernels of a specified layer of a CNN network
    :param path: path to the folder containing the trained network (e.g. 'W:\\Projekte\\Fahrwerkdiagnose\\Datenanalyse\\CNN\\results\\2019_10_31_09h_10_ArchVar_Detrend_cv5\\e_13_k_064_cv5\\fold_0')
    :type path: string
    :param tensor_name: name of the kernel to plot (e.g. 'CNN_Model/Stage_0/conv_0/conv1D/kernel:0')
    :type tensor_name: string
    :return: nothing
    """
    # plot_kernels(path='W:\\Projekte\\Fahrwerkdiagnose\\Datenanalyse\\CNN\\results\\2019_10_31_09h_10_ArchVar_Detrend_cv5\\e_13_k_256_cv5\\fold_0', tensor_name='CNN_Model/Stage_0/conv_0/conv1D/kernel:0')

    # change to specified folder
    os.chdir(path)

    # number_of_readin_rows = 42
    # config = dict()
    # with open(path+os.sep+'log.txt', 'rt') as file:
    #     line = [next(file) for x in range(number_of_readin_rows)]
    # values = [x.split(' ') for x in line]
    #
    # for row in values
    # [config[x[0]] = x[1] for x in values]

    file = open(path+os.sep+'log.txt', 'rt')
    # datafile = file('example.txt')
    # found = False
    config = dict()
    for line in file:
        line.split(' ')
        if 'sensors ' in line and 'avail_sensors' not in line:
            sel_sensor_names = line[line.find("['")+2:line.find("']")]
            sel_sensor_names = sel_sensor_names.split("', '")
            break

    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('metagraph.meta')
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name(tensor_name).eval()
        # fc = graph.get_tensor_by_name('CNN_Model/fc1_relu_bn/FC/kernel:0').eval()

    if 'fc' in locals():
        freq = np.arange(0, 50, 50 / 256)
        plt.figure()
        for cnt_kernel in range(0, fc.shape[1]):
            plt.figure()
            for cnt_signal in range(0, 7):
                plt.plot(freq, fc[cnt_signal*256:(cnt_signal+1)*256, cnt_kernel], label=sel_sensor_names[cnt_signal])
            plt.title('FC Kernel '+ str(cnt_kernel))
            plt.xlabel('Frequency in Hz')
            plt.ylabel('Weight')
            plt.legend()
        if show_fig:
            plt.show()

    if '2D' in tensor_name:
        # for cnt_kernel in range(0, x.shape[3]):
        #     v_max = np.amax(np.abs(x[:, :, :, cnt_kernel]))
        #     v_min = np.amin(np.abs(x[:, :, :, cnt_kernel]))
        #     # plt.figure()
        #     fig, axes = plt.subplots(2, 4, sharex='col', sharey='row')
        #     for cnt_signal in range(0, x.shape[2]):
        #         cnt_row = 0 if cnt_signal <= 3 else 1
        #         cnt_col = cnt_signal if cnt_signal <= 3 else cnt_signal-4
        #         axes[cnt_row, cnt_col].imshow(x[:, :, cnt_signal, cnt_kernel], vmin=v_min, vmax=v_max, cmap='gray_r')
        #         # axes[cnt_row, cnt_col].imshow(x[:, :, cnt_signal, cnt_kernel], cmap='gray_r')
        #         # axes[cnt_row, cnt_col].colorbar()
        #         # axes[cnt_row, cnt_col].title(sel_sensor_names[cnt_signal])
        #         # plt.subplots(2, 4, cnt_signal)
        #         # plt.imshow(x[:, :, cnt_signal, cnt_kernel], vmin=0, vmax=v_max, cmap='gray_r')
        #     if save_fig:
        #         path_to_save = path + os.sep + '_'.join(re.split('[/ :]', tensor_name))
        #         if not os.path.isdir(path_to_save):
        #             os.makedirs(path_to_save)  # make sure directory exists for writing the file
        #         plt.savefig(path_to_save + os.sep + str(cnt_kernel) + '.pdf')
        #         if export2tikz:
        #             tikz_save(path_to_save + os.sep + str(cnt_kernel) + '.tikz', encoding='utf8', show_info=False)
        # if show_fig:
        #     plt.show()

        # plot one specific kernel and one specific signal
        cnt_kernel = 15
        cnt_signal = 3
        # v_max = np.amax(np.abs(x[:, :, :, cnt_kernel]))
        # v_max = np.amax(np.abs(x))
        v_max = np.amax(x)
        v_min = np.amin(x)
        plt.figure()
        plt.imshow(x[:, :, cnt_signal, cnt_kernel], vmin=v_min, vmax=v_max, cmap='gray_r')
        plt.colorbar()
        plt.xlabel('Index of Receptive Field (Time)')
        plt.ylabel('Index of Receptive Field (Frequency)')
        plt.title(sel_sensor_names[cnt_signal])
        if save_fig:
            path_to_save = path + os.sep + '_'.join(re.split('[/ :]', tensor_name))
            if not os.path.isdir(path_to_save):
                os.makedirs(path_to_save)  # make sure directory exists for writing the file
            plt.savefig(path_to_save + os.sep + 'kernel' + str(cnt_kernel) + '_' + sel_sensor_names[cnt_signal] + '.pdf')
            if export2tikz:
                tikz_save(path_to_save + os.sep + 'kernel' + str(cnt_kernel) + '_' + sel_sensor_names[cnt_signal] + '.tikz', encoding='utf8', show_info=False)

        # plot one specific kernel and one specific signal
        cnt_kernel = 1
        cnt_signal = 3
        # v_max = np.amax(np.abs(x[:, :, :, cnt_kernel]))
        # v_max = np.amax(np.abs(x))
        v_max = np.amax(x)
        v_min = np.amin(x)
        plt.figure()
        plt.imshow(x[:, :, cnt_signal, cnt_kernel], vmin=v_min, vmax=v_max, cmap='gray_r')
        plt.colorbar()
        plt.xlabel('Index of Receptive Field (Time)')
        plt.ylabel('Index of Receptive Field (Frequency)')
        plt.title(sel_sensor_names[cnt_signal])
        if save_fig:
            path_to_save = path + os.sep + '_'.join(re.split('[/ :]', tensor_name))
            if not os.path.isdir(path_to_save):
                os.makedirs(path_to_save)  # make sure directory exists for writing the file
            plt.savefig(path_to_save + os.sep + 'kernel' + str(cnt_kernel) + '_' + sel_sensor_names[
                cnt_signal] + '.pdf')
            if export2tikz:
                tikz_save(path_to_save + os.sep + 'kernel' + str(cnt_kernel) + '_' + sel_sensor_names[
                    cnt_signal] + '.tikz', encoding='utf8', show_info=False)

    if '1D' in tensor_name:
        for cnt_kernel in range(0, x.shape[2]):
            plt.figure()    # plot one figure for each kernel
            for cnt_signal in range(0, x.shape[1]):
                plt.subplot(2, 1, 1)
                plt.plot(x[:, cnt_signal, cnt_kernel], label=sel_sensor_names[cnt_signal])   # plot the weights of all signals into one figure

                # perform FFT for kernel weights
                # data = x[:, cnt_signal, cnt_kernel]
                # time_step = 0.01
                # fs = 1 / time_step
                # freqs = np.linspace(0.0, N * time_step, N)
                # freqs = np.fft.fftfreq(data.size, time_step)  # array of frequencies
                # ps = np.abs(np.fft.fft(data)) ** 2  # array of power spectrum

                # plot FFT analysis in subplot
                plt.subplot(2, 1, 2)
                plt.psd(x[:, cnt_signal, cnt_kernel], Fs=1/0.01)
                # plt.plot(freqs, ps)

            plt.subplot(2, 1, 1)
            plt.xlabel('Index of Receptive Field')
            plt.ylabel('Weight')
            plt.title(tensor_name + ' - ' + str(cnt_kernel))

            # plt.subplot(2, 1, 2)
            # plt.xlabel('Index of Receptive Field')
            # plt.ylabel('Power of Frequency')
            # plt.title('Frequency Analysis of Kernel')

            if save_fig:
                path_to_save = path + os.sep + '_'.join(re.split('[/ :]', tensor_name))
                if not os.path.isdir(path_to_save):
                    os.makedirs(path_to_save)  # make sure directory exists for writing the file
                plt.savefig(path_to_save + os.sep + str(cnt_kernel) + '.pdf')
                if export2tikz:
                    tikz_save(path_to_save + os.sep + str(cnt_kernel) + '.tikz', encoding='utf8', show_info=False)
                
    if show_fig:
        plt.show()
