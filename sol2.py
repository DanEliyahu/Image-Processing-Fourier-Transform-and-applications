import math
import numpy as np
import scipy.io.wavfile
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from imageio import imread
from skimage.color import rgb2gray

GRAYSCALE = 1


def DFT(signal):
    """
    Compute the one-dimensional discrete Fourier Transform
    :param signal: the original signal as array of float64 with shape (N,1)
    :return: DFT of signal as array of complex128 with shape (N,1)
    """
    N = signal.shape[0]
    if N == 0:
        return signal
    # create grid for DFT matrix
    x = np.arange(N)
    DFT_matrix = np.exp((-2 * math.pi * 1j * x.reshape(N, 1) * x) / N)
    return signal.T.dot(DFT_matrix).T


def IDFT(fourier_signal):
    """
    Compute the one-dimensional inverse discrete Fourier Transform
    :param fourier_signal: the fourier signal as array of complex128 with shape (N,1)
    :return: the original signal as array of float64 with shape (N,1)
    """
    N = fourier_signal.shape[0]
    if N == 0:
        return fourier_signal
    # create grid for IDFT matrix
    x = np.arange(N)
    IDFT_matrix = np.exp((2 * math.pi * 1j * x.reshape(N, 1) * x) / N) / N
    return np.real_if_close(fourier_signal.T.dot(IDFT_matrix).T).astype(np.complex128)


def DFT2(image):
    """
    Compute the 2-dimensional discrete Fourier Transform
    :param image: grayscale image of float64 with shape (M,N,1)
    :return: DFT of image as array of complex128 with shape (M,N,1)
    """
    # M, N = image.shape[:2]
    # image = image.astype(np.complex128)
    # for row in range(M):  # DFT over rows
    #     image[row] = DFT(image[row])
    #
    # for col in range(N):  # DFT over cols
    #     image[:, col] = DFT(image[:, col])
    #
    # return image

    # apply DFT matrix on all cols and then DFT matrix again on all rows
    # swapaxes is used to handle case of (M,N,1) where transpose doesn't work as needed
    return DFT(DFT(image).swapaxes(0, 1)).swapaxes(0, 1)


def IDFT2(fourier_image):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform
    :param fourier_image: 2D array of complex128 with shape (M,N,1)
    :return: the original image as array of float64 with shape (M,N,1)
    """
    # M, N = fourier_image.shape[:2]
    # image = fourier_image.astype(np.complex128)
    # for row in range(M):  # DFT over rows
    #     image[row] = IDFT(image[row])
    #
    # for col in range(N):  # DFT over cols
    #     image[:, col] = IDFT(image[:, col])
    #
    # return np.real_if_close(image).astype(np.complex128)

    # apply IDFT matrix on all cols and then IDFT matrix again on all rows
    # swapaxes is used to handle case of (M,N,1) where transpose doesn't work as needed
    return np.real_if_close(IDFT(IDFT(fourier_image).swapaxes(0, 1))).swapaxes(0, 1).astype(np.complex128)


def change_rate(filename, ratio):
    """
    Changes the duration of an audio file by changing the sampling rate
    :param filename: string representing the path to a WAV file
    :param ratio: positive float64 representing the duration change
    """
    rate, data = scipy.io.wavfile.read(filename)
    scipy.io.wavfile.write('change_rate.wav', int(rate * ratio), data)


def change_samples(filename, ratio):
    """
    Changes the duration of an audio file by reducing the number of samples using Fourier
    :param filename: string representing the path to a WAV file
    :param ratio: positive float64 representing the duration change
    :return: 1D ndarray of dtype float64 representing the new sample points
    """
    rate, data = scipy.io.wavfile.read(filename)
    data = data.astype(np.float64)
    # data = data / np.iinfo(np.int16).max  # uncomment if want to save as normalized values for audio to make sense
    data = resize(data, ratio)
    scipy.io.wavfile.write('change_samples.wav', rate, data)
    return data


def resize_spectrogram(data, ratio):
    """
    Speeds up a WAV file, without changing the pitch, using spectrogram scaling
    :param data: 1D ndarray of dtype float64 representing the original sample points,
    :param ratio: positive float64 representing the rate change of the WAV file
    :return: new sample points according to ratio with the same datatype as data
    """
    spectrogram = stft(data)
    new_data = []
    for row in range(spectrogram.shape[0]):
        new_data.append(resize(spectrogram[row], ratio))
    new_data = np.vstack(new_data)
    new_data = istft(new_data)
    return new_data


def resize_vocoder(data, ratio):
    """
    Speeds up a WAV file by phase vocoding its spectrogram
    :param data: 1D ndarray of dtype float64 representing the original sample points,
    :param ratio: positive float64 representing the duration change
    :return: data rescaled according to ratio with the same data type as data
    """
    spectrogram = stft(data)
    new_data = phase_vocoder(spectrogram, ratio)
    return istft(new_data)


def resize(data, ratio):
    """
    Changes the number of samples by the given ratio
    :param data: 1D ndarray of dtype float64 or complex128 representing the original sample points
    :param ratio: positive float64 representing the duration change
    :return: 1D ndarray of the dtype of data representing the new sample points
    """
    if ratio == 1:
        return data
    data_type = data.dtype
    data = DFT(data)
    data = np.fft.fftshift(data)

    delta_samples = abs(data.size - math.floor(data.size / ratio))  # number of samples to add or remove
    if ratio > 1:  # speed up, remove samples
        data = data[math.floor(delta_samples / 2):data.size - math.ceil(delta_samples / 2)]  # clip data
    else:  # slow down, add padding samples
        left_pad = np.zeros(math.floor(delta_samples / 2))
        right_pad = np.zeros(math.ceil(delta_samples / 2))
        data = np.append(left_pad, np.append(data, right_pad))

    data = np.fft.ifftshift(data)
    data = IDFT(data)
    data = data.astype(data_type)
    return data


def conv_der(im):
    """
    Computes the magnitude of image derivatives using convolution
    :param im: grayscale image of type float64
    :return: magnitude of the derivative, with the same dtype and shape
    """
    conv_vec = np.array([[0.5, 0, -0.5]])
    dx = scipy.signal.convolve2d(im, conv_vec, mode='same')
    dy = scipy.signal.convolve2d(im, conv_vec.reshape(3, 1), mode='same')
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


# noinspection DuplicatedCode
def fourier_der(im):
    """
    Computes the magnitude of image derivatives using Fourier transform
    :param im: grayscale image of type float64
    :return: magnitude of the derivative, with the same dtype and shape
    """
    rows, cols = im.shape
    fourier_transform = np.fft.fftshift(DFT2(im))

    # get dx by making [-N/2,N/2] coef vec
    x_coef = np.arange(int(-cols / 2), math.ceil(cols / 2)) * ((2 * np.pi * 1j) / cols)
    dx = fourier_transform * x_coef
    dx = IDFT2(np.fft.ifftshift(dx))

    # get dy by making [-M/2, M/2] coef vec
    y_coef = np.arange(int(-rows / 2), math.ceil(rows / 2)) * ((2 * np.pi * 1j) / rows)
    dy = fourier_transform * y_coef[:, None]  # make y_coef col vector
    dy = IDFT2(np.fft.ifftshift(dy))

    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


#  - - - - - - - - - - added functions from helper and prev ex- - - - - - - - - - - - - -
def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should be a
    grayscale image (1) or an RGB image (2)
    :return: the image represented by a matrix of type np.float64
    """
    im = imread(filename)
    if representation == GRAYSCALE and im.ndim == 3:
        im = rgb2gray(im)
        return im
    im = im.astype(np.float64)
    im /= 255
    return im


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
