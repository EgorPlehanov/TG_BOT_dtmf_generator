import telebot
from typing import Any
from decouple import config
import re

from dtmf import DTMF

API_TOKEN = config('BOT_TOKEN')
bot = telebot.TeleBot(API_TOKEN)
bot_chat_parameters = {}


def create_chat_parameters(chat_id) -> dict:
    '''
    Создание параметров для чата

    mode - режим работы бота
    mode values:
        "processing" - режим обработки DTMF сигнала (генерация и декодирование)
        "settings" - режим настроек параметров генератора DTMF
        f"settings|{parameter_name}" - режим настроеки выбранного праметра генератора DTMF
    '''
    global bot_chat_parameters
    bot_chat_parameters[chat_id] = {
        'mode': 'processing',
        'dtmf': DTMF()
    }
    
    return bot_chat_parameters[chat_id]


def get_chat_parameter(chat_id, parameter_name) -> Any:
    '''Получение параметра для чата'''
    global bot_chat_parameters

    chat_paremeters = bot_chat_parameters.get(chat_id)
    if not chat_paremeters:
        chat_paremeters = create_chat_parameters(chat_id)
    
    return chat_paremeters.get(parameter_name)


def set_chat_parameter(chat_id, parameter_name, parameter_value) -> None:
    '''Установка параметра для чата'''
    global bot_chat_parameters

    chat_paremeters = bot_chat_parameters.get(chat_id)
    if not chat_paremeters:
        chat_paremeters = create_chat_parameters(chat_id)
    
    chat_paremeters[parameter_name] = parameter_value



@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    '''Отправляет приветственное сообщение'''
    create_chat_parameters(message.chat.id)

    bot.reply_to(message, (
        f'Привет, {message.from_user.first_name}!\n\n'
        'Я - бот для генерации и распознавания DTMF сигнала. 🤖\n\n'
        '📞 Для генерации DTMF сигнала отправьте мне номер телефона. Допустимые символы: 0-9, *, # и A-D.\n\n'
        '🎙️ Для распознавания DTMF сигнала отправьте мне голосовое сообщение.\n\n'
        '📚 Отправьте /info, чтобы узнать больше о DTMF и как его использовать.\n\n'
        '⚙️ Задать кастомные параметры генератора /settings.'
    ))


def get_file(file_path, mode='r', encoding='utf-8') -> str:
    '''Возвращает содержимое файла'''
    try:
        with open(file_path, mode, encoding=encoding) as file:
            return file.read()
    except Exception as e:
        print(e)
        return 'Файл не найден.'


@bot.message_handler(commands=['info'])
def send_info(message):
    '''Отправляет общую информацию о DTMF'''
    dtmf_info = get_file("src\\text\info_dtmf.txt")
    bot.send_message(message.chat.id, dtmf_info, parse_mode='HTML')

    dtmf_frequencies_photo = get_file("src\img\dtmf_frequencies.png", 'rb', None)
    bot.send_photo(message.chat.id, photo=dtmf_frequencies_photo, caption='Частоты DTMF сигнала')


@bot.message_handler(commands=['info_dtmf_generate_signal'])
def send_info_dtmf_generate_signal(message):
    '''Отправляет информацию о генерации DTMF сигнала'''
    dtmf_info = get_file("src\\text\info_dtmf_generate_signal.txt")
    bot.send_message(message.chat.id, dtmf_info, parse_mode='HTML')


@bot.message_handler(commands=['info_dtmf_recognition_signal'])
def send_info_dtmf_recognition_signal(message):
    '''Отправляет информацию о распознавании DTMF сигнала'''
    dtmf_info = get_file("src\\text\info_dtmf_recognition_signal.txt")
    bot.send_message(message.chat.id, dtmf_info, parse_mode='HTML')



@bot.message_handler(commands=['settings'])
def send_settings(message):
    '''Возвращает режим настроек параметров генератора DTMF'''
    set_chat_parameter(message.chat.id, 'mode', "settings")

    bot.send_message(
        message.chat.id,
        '‼️ Бот в режиме настроек параметров генератора DTMF ⚙️'
    )
    
    dtmf = get_chat_parameter(message.chat.id, 'dtmf')
    dtmf_parameters = dtmf.get_parameters()

    bot.send_message(
        message.chat.id,
        (
            '🔧 Настройки параметров генератора: \n\n' +
            '\n'.join([
                f"{parameter_data.get('name', 'Название отсутствует')}: {parameter_data.get('value')} {parameter_data.get('unit', '')}"
                for _, parameter_data in dtmf_parameters.items()
            ]) +
            '\n\n⬇️Нажми на нужный параметр для изменения⬇️'
        ),
        reply_markup=telebot.util.quick_markup({
            **{
                parameter_data.get('name', 'Название отсутствует'): {'callback_data': parameter_name}
                for parameter_name, parameter_data in dtmf_parameters.items()
            },
            'Выйти из настроек': {'callback_data': 'cancel'}
        })
    )



@bot.callback_query_handler(
    func=lambda call: "settings" in get_chat_parameter(call.message.chat.id, 'mode')
)
def callback_query_settings(call):
    '''Обработка нажатия на кнопку в режиме настроек параметров генератора DTMF'''
    if call.data == 'cancel':
        set_chat_parameter(call.message.chat.id, 'mode', "processing")
        bot.send_message(
            call.message.chat.id,
            '‼️ Бот в режиме генерации и распознавания DTMF сигнала 🎛️',
        )
        return
    if call.data == 'cancel_set_parameter':
        set_chat_parameter(call.message.chat.id, 'mode', "settings")
        bot.delete_message(call.message.chat.id, call.message.message_id)
        return

    dtmf = get_chat_parameter(call.message.chat.id, 'dtmf')
    dtmf_parameters = dtmf.get_parameters()
    parameter_name = dtmf_parameters.get(call.data).get('name')
    parameter_unit = dtmf_parameters.get(call.data).get('unit')

    bot.send_message(
        call.message.chat.id,
        (
            "Отправьте значение для параметра:\n"
            f"{parameter_name} {('('+ parameter_unit +')' if parameter_unit else '')}"
        ),
        reply_markup=telebot.util.quick_markup({
            'Отмена': {'callback_data': 'cancel_set_parameter'}
        })
    )

    set_chat_parameter(call.message.chat.id, 'mode', f"settings|{call.data}")



@bot.message_handler(
    content_types=['text'],
    func=lambda message: 'settings|' in get_chat_parameter(message.chat.id, 'mode')
)
def handle_parameter_text(message):
    '''Обработка текстового сообщения в режиме настроек параметра генератора DTMF'''
    bot_mode = get_chat_parameter(message.chat.id, 'mode')
    parameter_name = bot_mode.split('|')[1]
    
    dtmf = get_chat_parameter(message.chat.id, 'dtmf')
    dtmf_parameters = dtmf.get_parameters()

    parameter_data = dtmf_parameters.get(parameter_name)
    parameter_converter = parameter_data.get('converter')
    parameter_validator = parameter_data.get('validator')
    parameter_print_name = parameter_data.get('name') or parameter_name

    try:
        parameter_value = parameter_converter(message.text)
        if not parameter_validator(parameter_value):
            raise ValueError("Недопустимое значение")
        
        dtmf.set_parameter(parameter_name, parameter_value)
        bot.send_message(
            message.chat.id,
            f"Параметр {parameter_print_name} успешно установлен в {parameter_value}"
        )
    except ValueError as e:
        bot.send_message(
            message.chat.id,
            f"Ошибка: {e}\nНеправильное значение для параметра {parameter_print_name}. Попробуйте ещё раз."
        )
    except Exception:
        bot.send_message(
            message.chat.id,
            f"Произошла ошибка при установке параметра {parameter_print_name}. Попробуйте ещё раз."
        )
    
    send_settings(message)
    


@bot.message_handler(
    content_types=['text'],
    func=lambda message: get_chat_parameter(message.chat.id, 'mode') == 'processing'
)
def handle_number_text(message):
    '''Возвращает сгенерированный DTMF сигнал по переданному номеру'''
    phone_number = message.text.strip().upper()

    if not re.match(r"^[0-9#*A-D]+$", phone_number):
        bot.reply_to(message, 'Напиши мне, пожалуйста, номер телефона,\nсостоящий из цифр и знаков # и *.')
        return

    dtmf = get_chat_parameter(message.chat.id, 'dtmf')
    signal_file, format = dtmf.get_dtmf_signal_file(phone_number, format="mp3")

    bot.send_audio(
        chat_id=message.chat.id,
        audio=signal_file,
        title=f"DTMF: {phone_number}",
        performer="DTMF Bot",
        reply_to_message_id=message.message_id,
        reply_markup=telebot.util.quick_markup({
            'Распознать сигнал': {'callback_data': 'voice_processing'}
        })
    )


@bot.callback_query_handler(
    func=lambda call: "processing" in get_chat_parameter(call.message.chat.id, 'mode')
)
def handle_recognize_signal_callback(call):
    '''Обработка нажатия на кнопку "Распознать сигнал"'''
    voice_processing(call.message)


@bot.message_handler(
    content_types=['audio', 'voice', 'document'],
    func=lambda message: get_chat_parameter(message.chat.id, 'mode') == 'processing'
)
def voice_processing(message):
    '''Возвращает распознанный номер и графика по полученному сигналу'''
    answer_message = bot.reply_to(message, f"Нужно подумать...")

    content_type_to_processing = {
        'audio': {
            'get_file_info': lambda message: bot.get_file(message.audio.file_id),
            'file_format': 'mp3'
        },
        'voice': {
            'get_file_info': lambda message: bot.get_file(message.voice.file_id),
            'file_format': 'ogg'
        },
        'document': {
            'get_file_info': lambda message: bot.get_file(message.document.file_id),
            'file_format': 'wav'
        }
    }

    bot_processing = content_type_to_processing[message.content_type]
    if not bot_processing:
        bot.delete_message(answer_message.chat.id, answer_message.message_id)
        bot.reply_to(message, f"Неподдерживаемый тип файла.")
        return
    
    file_info = bot_processing['get_file_info'](message)
    file_format = bot_processing['file_format']
            
    print(f"Файл: {file_info.file_path}") # DEBUG

    downloaded_file = bot.download_file(file_info.file_path)
    try:
        dtmf = get_chat_parameter(message.chat.id, 'dtmf')
        phone_number, images = dtmf.recognize_dtmf(downloaded_file, file_format)
    except Exception as e:
        bot.edit_message_text("Ошибка при распознавании.", answer_message.chat.id, answer_message.message_id)
        print(e)
        return

    if phone_number:
        bot.edit_message_text(
            f"Набран номер: {phone_number}",
            answer_message.chat.id,
            answer_message.message_id,
            disable_web_page_preview = True
        )
    else:
        bot.edit_message_text("Не удалось распознать номер.", answer_message.chat.id, answer_message.message_id)
    
    if images is not None:
        bot.send_media_group(
            answer_message.chat.id,
            [telebot.types.InputMediaPhoto(img) for img in images]
        )



@bot.message_handler(func=lambda message: True)
def echo_message(message):
    '''Обработка неизвестного сообщения'''
    if 'settings' in get_chat_parameter(message.chat.id, 'mode'):
        bot.reply_to(message, 'Нажми на кнопку параметра чтобы его выбрать')
    else:
        bot.reply_to(message, 'Я тебя не понимаю. Напиши /help.')



bot.infinity_polling()


# venv_dtmf\Scripts\activate
# python app\dtmf_tg_bot.py