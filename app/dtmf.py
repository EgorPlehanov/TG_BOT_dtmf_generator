import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from pydub import AudioSegment
from io import BytesIO

from dtmf_plotter import plot_dtmf_analysis


class DTMF:
    def __init__(self, duration=0.1, silence_duration=0.1, sampling_rate=8000, A=0.5):
        self.duration = duration
        self.silence_duration = silence_duration
        self.sampling_rate = sampling_rate
        self.A = A
        self.dtmf_frequencies = {
            '1': (697, 1209),
            '2': (697, 1336),
            '3': (697, 1477),
            'A': (697, 1633),

            '4': (770, 1209),
            '5': (770, 1336),
            '6': (770, 1477),
            'B': (770, 1633),

            '7': (852, 1209),
            '8': (852, 1336),
            '9': (852, 1477),
            'C': (852, 1633),

            '*': (941, 1209),
            '0': (941, 1336),
            '#': (941, 1477),
            'D': (941, 1633)
        }

    
    def get_parameters(self):
        return {
            'duration': {
                'name': '⌛ Длительность сигнала',
                'unit': 'секунд',
                'value': self.duration,
                'validator': lambda x: isinstance(x, (int, float)) and x >= 0,
                'converter': lambda x: float(x)
            },
            'silence_duration': {
                'name': '⌚ Интервал между сигналами',
                'unit': 'секунд',
                'value': self.silence_duration,
                'validator': lambda x: isinstance(x, (int, float)) and x >= 0,
                'converter': lambda x: float(x)
            },
            'sampling_rate': {
                'name': ' 🎶 Частота дискретизации',
                'unit': 'Гц',
                'value': self.sampling_rate,
                'validator': lambda x: isinstance(x, (int, float)) and x > 0,
                'converter': lambda x: int(x)
            },
            'A': {
                'name': ' 🔊 Амплитуда сигнала',
                'value': self.A,
                'validator': lambda x: isinstance(x, (int, float)) and x >= 0,
                'converter': lambda x: float(x)
            }
        }
    

    def set_parameter(self, parameter_name, parameter_value):
        setattr(self, parameter_name, parameter_value)


    def generate_dtmf_tone(self, phone_number):
        t = np.linspace(0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)

        dtmf_signal = []
        for digit in phone_number:
            f1, f2 = self.dtmf_frequencies[digit]

            signal_1 = self.A * np.sin(2 * np.pi * f1 * t)
            signal_2 = self.A * np.sin(2 * np.pi * f2 * t)
            dtmf_signal.extend(signal_1 + signal_2)

            silence = np.zeros(int(self.silence_duration * self.sampling_rate))
            dtmf_signal.extend(silence)

        dtmf_signal = np.int16(dtmf_signal / np.max(np.abs(dtmf_signal)) * 32767) # Normalize the signal to [-32767, 32767] and convert it to 16-bit integer

        return dtmf_signal
    

    def decode_dtmf(self, signal, ofset=10):
        window = self.duration + self.silence_duration

        # Initialize empty list to store the decoded keys and frequencies found
        keys = []
        found_freqs = []

        # Iterate through the signal in window-sized chunks
        for i in range(0, len(signal), int(self.sampling_rate * window)):
            # Get the current chunk of the signal
            cut_sig = signal[i:i+int(self.sampling_rate * window)]

            # Take the Fast Fourier Transform (FFT) of the current chunk
            fft_sig = np.fft.fft(cut_sig, self.sampling_rate)

            # Take the absolute value of the FFT
            fft_sig = np.abs(fft_sig)

            # Set the first 500 elements of the FFT to 0 (removes DC component)
            fft_sig[:500] = 0

            # Only keep the first half of the FFT (removes negative frequencies)
            fft_sig = fft_sig[:int(len(fft_sig)/2)]

            # Set the lower bound to be 75% of the maximum value in the FFT
            lower_bound = 0.75 * np.max(fft_sig)

            # Initialize empty list to store the frequencies that pass the lower bound threshold
            filtered_freqs = []

            # Iterate through the FFT and store the indices of the frequencies that pass the lower bound threshold
            for i, mag in enumerate(fft_sig):
                if mag > lower_bound:
                    filtered_freqs.append(i)

            # Iterate through the DTMF frequencies and check if any of the filtered frequencies fall within the expected range
            for char, frequency_pair in self.dtmf_frequencies.items():
                high_freq_range = range(
                    frequency_pair[0] - ofset, frequency_pair[0] + ofset + 1)
                low_freq_range = range(
                    frequency_pair[1] - ofset, frequency_pair[1] + ofset + 1)
                if any(freq in high_freq_range for freq in filtered_freqs) and any(freq in low_freq_range for freq in filtered_freqs):
                    # If a match is found, append the key and frequency pair to the lists
                    keys.append(char)
                    found_freqs.append(frequency_pair)
        # Return the decoded keys and found frequencies
        return keys, found_freqs
    

    def play_dtmf_tone(self, number):
        signal = self.generate_dtmf_tone(number)
        sd.play(signal, self.sampling_rate)
        sd.wait()


    def save_dtmf_to_wav(self, number, filename):
        signal = self.generate_dtmf_tone(number)
        wavfile.write(filename, self.sampling_rate, signal)


    def get_dtmf_signal_file(self, number, format="wav"):
        signal = self.generate_dtmf_tone(number)

        if format == "wav":
            return signal, "wav"
        elif format == "mp3":
            audio = AudioSegment(signal.tobytes(), frame_rate=self.sampling_rate, sample_width=2, channels=1)
            output = BytesIO()
            audio.export(output, format="mp3")
            return output.getvalue(), "mp3"
        else:
            raise ValueError("Unsupported format")
        

    def recognize_dtmf(self, audio_data, file_format=None):
        audio = AudioSegment.from_file(BytesIO(audio_data), format=file_format)
        samples = np.array(audio.get_array_of_samples())

        keys, found_freqs = self.decode_dtmf(samples)
        images = plot_dtmf_analysis(samples, found_freqs, self.sampling_rate)
        return ''.join(keys), images
        


# if __name__ == '__main__':
#     dtmf = DTMF(duration=0.5)
#     phone_number = "12*#"  # Замените эту строку на нужный вам номер
#     output_file = "dtmf_signal.wav"  # Название выходного файла
#     # dtmf.play_dtmf_tone(phone_number)
#     # dtmf.save_dtmf_to_wav(phone_number, output_file)
#     # dtmf.get_dtmf_signal_file(phone_number, format="mp3")

#     wave_file = wave.open(output_file, 'r')
#     num_samples = wave_file.getnframes()
#     Fs = wave_file.getframerate()
#     data = wave_file.readframes(num_samples)

#     sample_width = wave_file.getsampwidth()

#     if sample_width == 1:
#         data = np.frombuffer(data, dtype=np.uint8)
#     elif sample_width == 2:
#         data = np.frombuffer(data, dtype=np.int16)

#     print(len(data))

#     wave_file.close()
#     print(dtmf.decode_dtmf(data))
