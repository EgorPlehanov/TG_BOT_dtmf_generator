import matplotlib.pyplot as plt
import numpy as np
import io

def save_plot_as_image(plt):
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    return img_buffer

def plot_dtmf_analysis(data, found_freqs, Fs):
    images = []

    # Создайте общий график с несколькими подграфиками
    plt.figure(figsize=(16, 9))

    # Постройте волну во временной области
    # plt.subplot(1, 1, 1)
    plt.plot(data)
    plt.xlabel('Индекс выборки')
    plt.ylabel('Амплитуда')
    plt.title('Волна во временной области')
    images.append(save_plot_as_image(plt))
    plt.clf()  # Очистите график для следующего подграфика

    # Постройте обнаруженные DTMF частоты во временной области
    # plt.subplot(2, 2, 2)
    for f1, f2 in found_freqs:
        plt.stem([f1, f2], [1, 1], 'k')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.title('Обнаруженные DTMF частоты во временной области')
    images.append(save_plot_as_image(plt))
    plt.clf()

    # Рассчитайте частотный спектр
    spectrum = np.abs(np.fft.fft(data))
    freqs = np.fft.fftfreq(len(data), 1/Fs)

    # Постройте частотный спектр
    # plt.subplot(2, 2, 3)
    plt.plot(freqs, spectrum)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.title('Частотный спектр')
    images.append(save_plot_as_image(plt))
    plt.clf()

    # Постройте обнаруженные DTMF частоты в частотной области
    # plt.subplot(2, 2, 4)
    for f1, f2 in found_freqs:
        idx1 = np.argmin(np.abs(freqs - f1))
        idx2 = np.argmin(np.abs(freqs - f2))
        plt.stem([freqs[idx1], freqs[idx2]], [
                 spectrum[idx1], spectrum[idx2]], 'k')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.title('Обнаруженные DTMF частоты в частотной области')
    images.append(save_plot_as_image(plt))
    plt.clf()

    return images
