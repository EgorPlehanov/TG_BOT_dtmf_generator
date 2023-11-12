import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from pydub import AudioSegment
from io import BytesIO
import os
from typing import Optional, List
from scipy.signal import butter, lfilter

from plotter import plot_dtmf_analysis_results


class DTMF:
    _NORMALIZATION_FACTOR = 32767
    _WINDOW_SIZE_MS = 20

    def __init__(
        self,
        duration = 0.1,                 # Длительность сигнала
        silence_duration = 0.05,        # Длительность паузы между сигналами
        sampling_rate = 16000,          # Частота дискретизации
        A = 0.5                         # Амплитуда сигнала
    ):
        self.duration = duration
        self.silence_duration = silence_duration
        self.sampling_rate = sampling_rate
        self.A = A
        
        self.symbols = ['1', '2', '3', 'A', '4', '5', '6', 'B', '7', '8', '9', 'C', '*', '0', '#', 'D']
        self.frequencies = [697, 770, 852, 941, 1209, 1336, 1477, 1633]
        self.low_frequencies = self.frequencies[:4]
        self.high_frequencies = self.frequencies[4:]
        
        self.symbols_to_frequencies = {
            symbol: (self.low_frequencies[index // 4], self.high_frequencies[index % 4])
            for index, symbol in enumerate(self.symbols)
        }
        self.frequencies_to_symbols = {
            frequencies: symbol
            for symbol, frequencies in self.symbols_to_frequencies.items()
        }


    def set_parameter(self, parameter_name: str, parameter_value: [int, float]) -> None:
        '''Устанавливает параметр в переданное значение'''
        setattr(self, parameter_name, parameter_value)

    
    def get_parameters(self) -> dict:
        '''Возвращает словарь с параметрами DTMF'''
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
    

    def play_dtmf_tone(self,
        signal: Optional[np.ndarray] = None,
        number: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> None:
        '''Проигрывает аудио сигнал DTMF'''
        if signal is None:
            if number is not None:
                signal = self.generate_dtmf_tone(number)
            elif file_path is not None:
                with open(file_path, 'rb') as f:
                    signal = np.frombuffer(f.read(), dtype=np.int16)
            return
                
        sd.play(signal, self.sampling_rate)
        sd.wait()


    def save_dtmf_to_wav(self,
        filename: str = "dtmf",
        signal: Optional[np.ndarray] = None,
        number: Optional[str] = None
    ) -> str:
        '''Сохраняет аудио сигнал DTMF в формате wav'''
        if signal is None:
            if number is not None:
                signal = self.generate_dtmf_tone(number)
            return
        
        output_path = os.path.join("src\\audio", f"{filename}.wav")
        wavfile.write(output_path, self.sampling_rate, signal)

        return output_path


    def generate_dtmf_tone(self, phone_number: str) -> np.ndarray:
        '''Генерирует аудио сигнал DTMF'''
        t = np.linspace(0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)

        dtmf_signals = [
            self.A * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))
            for digit in phone_number
            for f1, f2 in [self.symbols_to_frequencies[digit]]
        ]

        silence_duration_samples = int(self.silence_duration * self.sampling_rate)
        silence = np.zeros(silence_duration_samples)

        dtmf_signals_with_silence = np.concatenate([
            value
            for signal_silence in zip(dtmf_signals, [silence] * len(dtmf_signals))
            for value in signal_silence
        ][:-1])

        # Normalize the signal to [-32767, 32767] and convert it to 16-bit integer
        return np.int16(dtmf_signals_with_silence / np.max(np.abs(dtmf_signals_with_silence)) * self._NORMALIZATION_FACTOR)


    def get_dtmf_signal_file(self, number: str, format: str = "wav") -> tuple:
        '''Генерирует аудио сигнал DTMF и возвращает его в формате wav или mp3'''
        signal = self.generate_dtmf_tone(number)

        # file_path = self._save_dtmf_to_wav(number, signal) # Сохраняем сигнал в формате wav
        # self._play_dtmf_tone(file_path=file_path) # Проигрываем сигнал

        if format == "wav":
            return signal, "wav"
        elif format == "mp3":
            audio = AudioSegment(signal.tobytes(), frame_rate=self.sampling_rate, sample_width=2, channels=1)
            output = BytesIO()
            audio.export(output, format="mp3")
            return output.getvalue(), "mp3"
        else:
            raise ValueError("Unsupported format")
        

    def recognize_dtmf(self, audio_data: bytes, file_format: str = None) -> tuple:
        '''Распознавание сигнала DTMF'''
        audio = AudioSegment.from_file(BytesIO(audio_data), format=file_format)
        signal = np.array(audio.get_array_of_samples())
        sampling_rate = audio.frame_rate

        symbols, frequencies = self._decode_dtmf(signal, sampling_rate)
        graphs_images = plot_dtmf_analysis_results(signal, sampling_rate, frequencies, symbols)
        
        return ''.join(symbols), graphs_images
    

    def _goertzel(self, samples: np.ndarray, frequency: int | float, sample_rate: int) -> float:
        '''Алгоритм Гёрцеля'''
        sample_length = len(samples)

        coeff = 2 * np.cos(2 * np.pi * frequency / sample_rate)
        prev_1, prev_2 = 0, 0

        for n in range(sample_length):
            curr = samples[n] + coeff * prev_1 - prev_2
            prev_2 = prev_1
            prev_1 = curr
        
        return np.sqrt(curr**2 + prev_1**2)


    def _decode_dtmf(self, audio_data: np.ndarray, sample_rate: int) -> tuple:
        '''Распознавание сигнала DTMF'''
        decoded_numbers = []     # Распознанные символы
        decoded_frequencies = [] # Распознанные частоты

        audio_data = audio_data / max(abs(audio_data)) # Нормализация синала

        # Определение областей для декодирования сигнала
        chunks = self._get_chanks(audio_data, sample_rate)

        audio_data = self._filter_signal_by_frequencies(audio_data, sample_rate, self.frequencies)

        for start, end in chunks:
            chunk = audio_data[start:end]

            # Определение оценокт низких, высоких частот в сэмпле сигнала
            dtmf_low_freq_scores = self._calculate_gortzel_scores_for_frequencies(chunk, self.low_frequencies, sample_rate)
            dtmf_high_freq_scores = self._calculate_gortzel_scores_for_frequencies(chunk, self.high_frequencies, sample_rate)

            # Определение частот сигнала по максимальной оценке
            low_freq = self._get_frequency_by_index_max_score(dtmf_low_freq_scores, self.low_frequencies)
            high_freq = self._get_frequency_by_index_max_score(dtmf_high_freq_scores, self.high_frequencies)

            decoded_numbers.append(self.frequencies_to_symbols[(low_freq, high_freq)])

            decoded_frequencies.append({
                'symbol': self.frequencies_to_symbols[(low_freq, high_freq)],
                'low_frequency': low_freq,
                'high_frequency': high_freq,
                'position_start': start,
                'position_end': end
            })

        return decoded_numbers, decoded_frequencies
    

    def _calculate_gortzel_scores_for_frequencies(
        self,
        chunk: np.ndarray,
        frequencies: List[float],
        sample_rate: int
    ) -> List[float]:
        '''Вычисление оценок для каждой частоты алгоритмом Гёрцеля'''
        scores = [0] * len(frequencies)
        for freq_index, freq in enumerate(frequencies):
            scores[freq_index] = self._goertzel(chunk, freq, sample_rate)
        return scores
    

    def _get_frequency_by_index_max_score(self, scores: List[float], frequencies: List[float]) -> float:
        '''Определение частоты по индексу максимальной оценки'''
        max_score_index = scores.index(max(scores))
        return frequencies[max_score_index]
    

    def _get_chanks(self, audio_data: np.ndarray, sample_rate: int) -> List[tuple]:
        '''Разделение сигнала на фрагменты'''
        freq_filtered_signal = self._filter_signal_by_frequencies(audio_data, sample_rate, self.frequencies)
        amp_threshold = self._calculate_amplitude_threshold(freq_filtered_signal)
        amp_filtered_signal = self._amplitude_filter(freq_filtered_signal, amp_threshold)
        window_size = int(sample_rate * self._WINDOW_SIZE_MS / 1000)
        signal_energy = self._calculate_energy(amp_filtered_signal, window_size)
        chunks = self._get_energy_intervals(signal_energy)
        return chunks
    

    def _calculate_offset(self, center_frequency: float | int) -> int:
        '''Вычисляет смещение на основе центральной частоты'''
        if center_frequency >= 300 and center_frequency < 1000:
            return 20
        elif center_frequency >= 1000 and center_frequency < 3000:
            return 50
        elif center_frequency >= 3000 and center_frequency < 10000:
            return 200
        return 0


    def _butter_bandpass_filter(
        self,
        signal: np.ndarray,
        center_frequency: float,
        sample_rate: float,
        order: int = 4
    ) -> np.ndarray:
        '''Применяет полосовой фильтр Баттерворта к входному сигналу'''
        nyquist = 0.5 * sample_rate
        offset = self._calculate_offset(center_frequency)
        low = (center_frequency - offset) / nyquist
        high = (center_frequency + offset) / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, signal)
        return y


    def _filter_signal_by_frequencies(
        self,
        signal: np.ndarray,
        sample_rate: int,
        frequencies: List[float],
    ) -> np.ndarray:
        """Фильтрует сигнал по заданным частотам с использованием фильтра Баттерворта-полосы"""
        filtered_signal = np.sum([self._butter_bandpass_filter(signal, freq, sample_rate) for freq in frequencies], axis=0)
        return filtered_signal


    def _calculate_amplitude_threshold(self, signal: np.ndarray, multiplier: float = 1.0) -> float:
        """Вычисляет пороговое значение амплитуды."""
        mean_amplitude = np.mean(np.abs(signal))
        std_amplitude = np.std(np.abs(signal))
        threshold = mean_amplitude + multiplier * std_amplitude
        return threshold


    def _amplitude_filter(self, signal: np.ndarray, threshold: float) -> np.ndarray:
        '''Фильтрует сигнал по амплитуде'''
        filtered_signal = np.where(np.abs(signal) > threshold, signal, 0)
        return filtered_signal
    
    def _calculate_energy(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        '''Вычисляет энергию сигнала'''
        squared_signal = np.square(signal)
        energy = np.convolve(squared_signal, np.ones(window_size)/window_size, mode='valid')
        return energy
    

    def _get_energy_intervals(self, energy_values: np.ndarray) -> List[tuple]:
        """Находит интервалы, где энергия сигнала больше 0."""
        intervals = []
        start_index = None

        for i, energy in enumerate(energy_values):
            if energy > 0:
                if start_index is None:
                    start_index = i
            elif start_index is not None:
                intervals.append((start_index, i - 1))
                start_index = None

        if start_index is not None:
            intervals.append((start_index, len(energy_values) - 1))

        return intervals



if __name__ == '__main__':
    dtmf = DTMF(sampling_rate=8000)

    phone_number = "1122"
    signal = dtmf.generate_dtmf_tone(phone_number)
    result = dtmf.recognize_dtmf(signal, 'wav')
    print(result)