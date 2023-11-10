import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import numpy as np
from functools import partial


def convert_plot_to_image(plt)-> io.BytesIO:
    '''Конвертация графика в изображениеи представленное байтовой строкой'''
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.clf()
    buffer.seek(0)
    return buffer.read()


def plot_dtmf_analysis_results(
    audio_data,             # Исходный сигнал
    smpling_rate,           # Частота дискретизации
    found_freqs,            # Найденные частоты
    found_symbols,          # Найденные символы
    plot_names=None,        # Названия графиков
    graphic_size=(16, 9)    # Размер графика
) -> list:
    plot_functions = {
        'time_domain_signal': partial(
            get_plot_time_domain_signal,
            audio_data, smpling_rate, graphic_size
        ),
        'fft_spectrum': partial(
            get_plot_fft_spectrum,
            audio_data, smpling_rate, 0, 2000, graphic_size
        ),
        'spectrogram': partial(
            get_plot_spectrogram,
            audio_data, smpling_rate, graphic_size
        ),
        'dtmf_recognition_timeline': partial(
            get_plot_dtmf_recognition_timeline,
            audio_data, found_freqs, smpling_rate, found_symbols, graphic_size
        ),
    }
    if plot_names is None:
        plot_names = plot_functions.keys()

    plots = [
        plot_functions.get(name, lambda: None)()
        for name in plot_names
    ]
    images = [
        convert_plot_to_image(plot)
        for plot in plots if plot is not None
    ]
    return images


def get_plot_time_domain_signal(audio_data, sample_rate, graphic_size=(16, 9)):
    '''Временной график сигнала'''
    time = np.arange(0, len(audio_data)) / sample_rate

    plt.figure(figsize=graphic_size)
    plt.plot(time, audio_data)
    plt.title('Временной график аудиосигнала')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    return plt.gcf()


def get_plot_fft_spectrum(audio_data, sample_rate, min_freq=0, max_freq=20000, graphic_size=(16, 9)):
    '''Построение спектра БПФ'''
    fft_sig = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(fft_sig), 1/sample_rate)
    
    valid_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))
    valid_freqs = freqs[valid_indices]
    valid_fft_sig = np.abs(fft_sig[valid_indices])
    
    plt.figure(figsize=graphic_size)
    plt.plot(valid_freqs, valid_fft_sig)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.title('Спектр БПФ')
    plt.grid(True)
    return plt.gcf()


def get_plot_spectrogram(audio_data, sample_rate, graphic_size=(16, 9)):
    '''Построение спектрограммы'''
    plt.figure(figsize=graphic_size)
    plt.specgram(audio_data, Fs=sample_rate, cmap='inferno')
    plt.title('Спектрограмма аудиосигнала')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Частота (Гц)')
    plt.colorbar(label='Интенсивность')
    return plt.gcf()


def get_plot_dtmf_recognition_timeline(audio_data, detected_frequencies, sample_rate, detected_symbols, graphic_size=(16, 9)):
    '''Частотный график распознования во времени для алгоритма Гёрцеля'''
    plt.figure(figsize=graphic_size)

    for values in detected_frequencies:
        symbol = values.get('symbol')
        position_start = values.get('position_start')
        position_end = values.get('position_end')

        signal = audio_data[position_start:position_end]
        time_points = np.linspace(position_start/sample_rate, position_end/sample_rate, len(signal))

        plt.plot(time_points, signal, label=f'{symbol}')

    plt.title(f"Распознанные DTMF сигналы алгоритмом Гёрцеля во времени: {', '.join(detected_symbols)}")
    plt.xlabel('Время (секунды)')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True)
    return plt.gcf()
