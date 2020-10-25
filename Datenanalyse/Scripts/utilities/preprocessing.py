from sklearn import preprocessing
import numpy as np
from scipy.signal import detrend as detrend_fcn
from scipy.signal import spectrogram
from skimage import transform
from pywt import WaveletPacket
from pyts.image import RecurrencePlots
from pyts.image import GASF


def detrend(data, n_channels, seq_length, type='linear'):
    """apply detrend per sample per channel
    :param data the data to detrend
    :type data ndarray
    :param n_channels the number of channels
    :type n_channels int
    :param seq_length the length of a sequence of one single sensor
    :type seq_length int
    :param type the type of detrend to apply, 'linear' (default) or 'constant'
    :type type str
    :return data_detrend the detrended data array"""
    breakpoints = [seq_length * i for i in range(n_channels)]  # define breakpoints to separate channels
    data_detrend = np.zeros(data.shape)
    for sample in range(np.shape(data)[0]):     # loop over all training samples
        data_detrend[sample] = detrend_fcn(data[sample], bp=breakpoints, type=type)    # apply linear detrend, new detrend for every channel
    return data_detrend


def fourier(data, n_channels, seq_length):
    """ Compute the fft of a multichannel dataset
    :param data the raw data read from .csv or .npy file
    :type data ndarray
    :param n_channels number of sensors
    :type n_channels int
    :param seq_length length of one data stream of a single sensor
    :type seq_length int
    :return fft_data magnitude of half the windowed DFT spectrum
    :type fft_data ndarray
    :return fft_length new seq_length is one half of the original length
    :type fft_length int"""
    # Update seq_length due to symmetric spectrum
    fft_length = int(seq_length / 2)
    # initialize fft_data matrix
    fft_data = np.zeros((data.shape[0], fft_length*n_channels))
    # calc fft for each channel
    for ch in range(n_channels):
        fft_data[:, ch*fft_length:(ch+1)*fft_length] = fourier_transform(data[:, ch*seq_length:(ch+1)*seq_length])
    return fft_data, fft_length


def fourier_transform(x):
    """ Compute the row-wise fft of a matrix
    :param x matrix for which the DFT shall be computed
    :type x ndarray
    :return magnitude of half the windowed DFT spectrum
    :rtype ndarray"""
    # create hanning window to tackle spectral leakage
    hann = np.hanning(len(x[0, :]))
    # calc fft
    # out = np.fft.fft(hann*x)/len(hann)
    out = np.fft.fft(hann*x)/len(hann)
    # return only half of the symmetric spectrum
    new_length = int(len(out[0, :]) / 2)
    out = out[:, :new_length]
    return np.abs(out)


def concatenate_samples(data, n_channels, seq_length):
    """ Concatenate for each channel all samples
    :param data the data array with shape(n_samples x (n_channels*seq_length))
    :type data ndarray
    :param n_channels the number of channels
    :type n_channels int
    :param seq_length the length of one data stream of a single sensor
    :type seq_length int
    :returns datarow the data array with shape(n_channels x (n_samples*seq_length))
    """
    # separate channels from each other
    data_channels = np.reshape(data, [-1, n_channels, seq_length])
    # initialize result matrix
    datarow = np.zeros((n_channels, int(np.size(data, 0)*seq_length)))
    for ch in range(n_channels):
        # shape to one single row with all data points of the channel
        datarow[ch, :] = np.reshape(data_channels[:, ch, :], [1, -1])
    return datarow


def unconcatenate_samples(datarow, n_channels, seq_length):
    """ Reshapes concatenated samples to original data shape
    :param datarow the data array with shape(n_channels x (n_samples*seq_length))
    :type datarow ndarray
    :param n_channels number of channels
    :type n_channels int
    :param seq_length length of one data stream of a single sensor
    :type seq_length int
    :returns data data array with shape(n_samples x (n_channels*seq_length))
    :raises data ndarray
    """
    multichannel = []
    for ch in range(n_channels):
        # shape to one single row with all data points of the channel
        channel = np.reshape(datarow[ch, :], [-1, seq_length])
        multichannel.append(channel)
    data = np.concatenate([multichannel[i] for i in range(n_channels)], axis=1)  # create one array
    return data


def greyscale_image(data_flat, n_channels, seq_length):
    """
    :param data_flat: the raw data read from .csv or .npy file
    :param n_channels: number of channels
    :param seq_length: length of one data stream of a single sensor
    :return: greyscale images (16 bit), formatted the same way as the input data (rows=samples, columns= concatenated sensor signals)
    :return: tupel with image dimensions depending on the input stream length (recommended: even power of 2 --> no loss of data)
    """
    num_samples = np.shape(data_flat)[0]
    # separate channels
    data_channels = np.reshape(data_flat, [-1, n_channels, seq_length])
    # calculate size of grey scale image
    img_size = int(np.sqrt(seq_length))
    scaled_multichannel = []
    for ch in range(n_channels):
        # shape to one single row with all data points of the channel,
        # use only first img_size*img_size points of each sensor in case seq_length > img_size*img_size
        datarow = np.reshape(data_channels[:, ch, :img_size * img_size], [1, -1])
        scaled_row = np.round(preprocessing.minmax_scale(datarow, feature_range=(0, 65535), axis=1))  # scale to 16 bit
        scaled_channel = np.reshape(scaled_row, [num_samples, img_size * img_size])  # reshape to (samples, new_seqlength)
        scaled_multichannel.append(scaled_channel)
    scaled = np.concatenate([scaled_multichannel[i] for i in range(n_channels)], axis=1)  # create one array
    return scaled, [img_size, img_size]


def stft(dataset, n_channels, seq_length, img_size=32, downsample=True):
    """
    :param dataset: the raw data read from .csv or .npy file
    :param n_channels: number of channels
    :param seq_length: length of one data stream of a single sensor
    :return: the stft images, formatted the same way as the input data (rows=samples, columns= concatenated sensor signals)
    :return: tupel with image size
    """
    def specgram(sam, ch):
        f, t, Sxx = spectrogram(dataset[sam][ch * seq_length:(ch + 1) * seq_length],
                                # create spectrogram per sample per channel
                                detrend='linear',
                                # detrend=False,
                                nperseg=NFFT,
                                nfft=NFFT,
                                noverlap=48,
                                mode='magnitude',
                                scaling='spectrum'
                                )
        return f, t, Sxx

    NFFT = 64   # pain parameter to scale between frequency/time resolution
    num_samples = dataset.shape[0]
    if downsample:
        size_flat = img_size*img_size
    else:
        _, _, Sxx = specgram(0, 0)  # get the image dimensions if resizing is not applied
        size_flat = Sxx.shape[0] * Sxx.shape[1]
        img_h = Sxx.shape[0]
        img_w = Sxx.shape[1]
    # NFFT = seq_length // 4
    pics = np.zeros([num_samples, n_channels*size_flat])  # initialize array
    for sample in range(num_samples):
        for channel in range(n_channels):
            f, t, Sxx = specgram(sample, channel)
            if downsample:
                Sxx = transform.resize(Sxx, [img_size, img_size], mode='constant', anti_aliasing=0.15, preserve_range=True)
                img_h = img_w = img_size
            pics[sample][channel*size_flat:(channel+1)*size_flat] = np.reshape(np.flip(Sxx,0), [1, size_flat])
    return pics, [img_h, img_w]


def wpi(dataset, n_channels, seq_length):
    """
    function to calculate a wavelet packet energy image
    # Ding.2017: Energy-Fluctuated Multiscale Feature Learning With Deep ConvNet for Intelligent Spindle Bearing Fault Diagnosis
    :param dataset: the raw data read from .csv or .npy file
    :type dataset: ndarray
    :param n_channels: number of channels
    :type n_channels: int
    :param seq_length: length of one data stream of a single sensor
    :type seq_length: int
    :return: the flattened images, tupel holding the image size
    """
    level = 10  # choose an even number, paper uses 10
    wavelet = 'db8'    # Daubechies 8 used in paper
    # wavelet = 'coif3'  # Daubechies 8 used in paper
    order = "natural"  # other option is "freq"
    clip_energy = 1    # threshold to clip the calculated energy features (negative and positive)

    num_samples = dataset.shape[0]
    img_size = np.power(2, level // 2)
    size_flat = img_size*img_size
    pics = np.zeros([num_samples, n_channels * size_flat])  # initialize array
    wp_image = np.zeros([img_size, img_size])
    for sample in range(num_samples):   # loop over all samples
        for ch in range(n_channels):    # loop over all channels
            # Construct wavelet packet tree for signal of one channel
            wp = WaveletPacket(dataset[sample][ch*seq_length:(ch+1)*seq_length], wavelet, 'symmetric', maxlevel=level)
            nodes = wp.get_level(level, order=order)    # !required! access the tree to populate it (might be a bug of the library?)
            i = 0
            # loop through the tree (ordered from aaa..a to ddd..d)
            for node in wp.get_leaf_nodes():
                # use only the coefficients from node (i, p) with p = 0..(2^i-1), i.e. set all other coefficients to zero
                new_wp = WaveletPacket(None, wavelet, 'symmetric', level)
                new_wp[node.path] = wp[node.path].data
                # get the reconstruction coefficients (length 2^i)
                reconst = np.asarray(new_wp.reconstruct(update=False))
                # phase shift --> arrange energy features calculated as the squared sum over the reconstruction coefficients
                wp_image[i % img_size, i // img_size] = np.sum(np.multiply(reconst, reconst), axis=-1)
                # remove node from wp tree
                del new_wp[node.path]
                i += 1
            # (!) THP modification (!), clip the wpi to fixed range: especially the approximation coefficients hold a lot of energy which
            # scales very differently
            wp_image = np.clip(wp_image, -clip_energy, clip_energy)
            # collect all pictures, shape (samples, ch1+ch2..)
            pics[sample][ch*size_flat:(ch+1)*size_flat] = np.reshape(wp_image, [1, size_flat])
    return pics, [img_size, img_size]


def recurrence_plot(dataset, n_channels, seq_length, img_size=32, downsample=False):
    """
    function to transform the given dataset into recurrence plot images

    :param dataset: the raw data read from .csv or .npy file
    :type dataset: ndarray
    :param n_channels: number of channels
    :type n_channels: int
    :param seq_length: length of one data stream of a single sensor
    :type seq_length: int
    :param img_size: size of the recurrence plot image (if downsampling is enabled)
    :type img_size: int
    :param downsample: whether downsampling is applied or not
    :type downsample: bool
    :return: the flattened images, tupel holding the image size
    """
    rp = RecurrencePlots()
    if downsample:
        size_flat = img_size * img_size
    else:
        size_flat = seq_length * seq_length
    pics = np.zeros([dataset.shape[0], n_channels * size_flat])  # initialize array
    for ch in range(n_channels):  # loop over all channels
        pics_ch = rp.fit_transform(dataset[:, ch * seq_length:(ch + 1) * seq_length])   # calculate recurrence plot
        pics_ch = np.moveaxis(pics_ch, 0, 2)  # [samples, heigth, width] --> [heigth,width,samples]
        if downsample:
            pics_ch = transform.resize(pics_ch, [img_size, img_size], mode='constant', anti_aliasing=0.15,
                                       preserve_range=True)     # rescale (downsample)
        else:
            img_size = seq_length
        pics_ch = np.reshape(pics_ch, [size_flat, -1])  # [signal, samples]
        pics_ch = np.moveaxis(pics_ch, 0, 1)  # swap axes --> [samples, signal]
        pics[:, ch * size_flat:(ch + 1) * size_flat] = pics_ch
    return pics, [img_size, img_size]


def gasf_plot(dataset, n_channels, seq_length, img_size=32, downsample=False):
    """
    function to transform the given dataset into gramian angular summation field images
    see https://arxiv.org/pdf/1506.00327.pdf for details,
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8006280 for application example

    :param dataset: the raw data read from .csv or .npy file
    :type dataset: ndarray
    :param n_channels: number of channels
    :type n_channels: int
    :param seq_length: length of one data stream of a single sensor
    :type seq_length: int
    :return: the flattened images, tupel holding the image size
    """
    if downsample:
        size_flat = img_size * img_size
    else:
        size_flat = seq_length * seq_length
    gasf = GASF(int(np.sqrt(size_flat)))
    pics = np.zeros([dataset.shape[0], n_channels * size_flat])  # initialize array
    for ch in range(n_channels):  # loop over all channels
        pics_ch = gasf.fit_transform(dataset[:, ch * seq_length:(ch + 1) * seq_length])   # calculate recurrence plot
        pics_ch = np.moveaxis(pics_ch, 0, 2)  # [samples, heigth, width] --> [heigth,width,samples]
        if downsample:
            pics_ch = transform.resize(pics_ch, [img_size, img_size], mode='constant', anti_aliasing=0.15,
                                       preserve_range=True)     # rescale (downsample)
        else:
            img_size = seq_length
        pics_ch = np.reshape(pics_ch, [size_flat, -1])  # [signal, samples]
        pics_ch = np.moveaxis(pics_ch, 0, 1)  # swap axes --> [samples, signal]
        pics[:, ch * size_flat:(ch + 1) * size_flat] = pics_ch
    return pics, [img_size, img_size]