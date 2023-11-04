import telebot
from telebot.util import quick_markup
from decouple import config
import re

from dtmf import DTMF


API_TOKEN = config('BOT_TOKEN')
bot = telebot.TeleBot(API_TOKEN)
dtmf = DTMF()

# Определение режима работы бота
# "processing" - режим обработки DTMF сигнала (включая и генерацию, и декодирование)
# "settings" - режим настроек параметров
# f"settings_{parameter_name}" - режим настроеки выбранного праметра
bot_mode = "processing"  # Начальный режим



@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, (
        f'Привет, {message.from_user.first_name}!\n\n'
        'Я - бот для генерации и распознавания DTMF сигнала. 🤖\n\n'
        '📞 Для генерации DTMF сигнала отправьте мне номер телефона. Допустимые символы: 0-9, *, # и A-D.\n\n'
        '🎙️ Для распознавания DTMF сигнала отправьте мне голосовое сообщение.\n\n'
        '📚 Отправьте /info, чтобы узнать больше о DTMF и как его использовать.\n\n'
        '⚙️ Задать кастомные параметры генератора /settings.'
    ))



@bot.message_handler(commands=['info'])
def send_info(message):
    with open("src\\text\dtmf_info.txt", "r", encoding="utf-8") as file:
        dtmf_info = file.read()
        bot.send_message(
            message.chat.id,
            dtmf_info,
            parse_mode='HTML'
        )
    with open('src\img\dtmf_frequencies.png', 'rb') as dtmf_frequencies:
        bot.send_photo(
            message.chat.id,
            caption='Частоты DTMF',
            photo=dtmf_frequencies, 
        )



@bot.message_handler(commands=['settings'])
def send_settings(message):
    global bot_mode; bot_mode = "settings"
    bot.send_message(
        message.chat.id,
        '‼️ Бот в режиме настроек параметров генератора DTMF ⚙️'
    )

    dtmf_parameters = dtmf.get_parameters()
    markup_inline = telebot.util.quick_markup({
        **{
            parameter_data.get('name', 'Название отсутствует'): {'callback_data': parameter_name}
            for parameter_name, parameter_data in dtmf_parameters.items()
        },
        'Выйти из настроек': {'callback_data': 'cancel'}
    })
    bot.send_message(
        message.chat.id,
        (
            '🔧 Настройки параметров генератора: \n\n' +
            '\n'.join([
                f"{parameter_data.get('name', 'Название отсутствует')}: {parameter_data.get('value')} {parameter_data.get('unit', '')}"
                for parameter_name, parameter_data in dtmf_parameters.items()
            ]) +
            '\n\n⬇️Нажми на нужный параметр для изменения⬇️'
        ),
        reply_markup=markup_inline
    )



@bot.callback_query_handler(func=lambda call: "settings" in bot_mode)
def callback_query_settings(call):
    global bot_mode

    if call.data == 'cancel':
        bot_mode = "processing"
        bot.send_message(
            call.message.chat.id,
            '‼️ Бот в режиме генерации и распознавания DTMF сигнала 🎛️',
        )
        return
    if call.data == 'cancel_set_parameter':
        bot.delete_message(call.message.chat.id, call.message.message_id)
        bot_mode = "settings"
        return

    dtmf_parameters = dtmf.get_parameters()
    bot.send_message(
        call.message.chat.id,
        f"Отправьте значение для параметра:\n{dtmf_parameters.get(call.data).get('name')}",
        reply_markup=telebot.util.quick_markup({
            'Отмена': {'callback_data': 'cancel_set_parameter'}
        })
    )
    bot_mode = f"settings|{call.data}"



@bot.message_handler(
    content_types=['text'],
    func=lambda _: 'settings|' in bot_mode
)
def handle_parameter_text(message):
    global bot_mode
    parameter_name = bot_mode.split('|')[1]
    
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
    func=lambda _: bot_mode=='processing'
)
def handle_number_text(message):
    phone_number = message.text.strip().upper()

    if not re.match(r"^[0-9#*A-D]+$", phone_number):
        bot.reply_to(message, 'Напиши мне, пожалуйста, номер телефона,\nсостоящий из цифр и знаков # и *.')
        return

    signal_file, format = dtmf.get_dtmf_signal_file(phone_number, format="mp3")

    bot.send_audio(
        chat_id=message.chat.id,
        audio=signal_file,
        title=f"DTMF: {phone_number}",
        performer="DTMF Bot",
        reply_to_message_id=message.message_id
    )



@bot.message_handler(
    content_types=['audio', 'voice', 'document'],
    func=lambda _: bot_mode=='processing'
)
def voice_processing(message):
    answer_message = bot.reply_to(message, f"Нужно подумать...")

    file_info, file_format = None, None
    match message.content_type:
        case 'audio':
            file_info = bot.get_file(message.audio.file_id)
            file_format = 'mp3'
        case 'voice':
            file_info = bot.get_file(message.voice.file_id)
            file_format = 'ogg'
        case 'document':
            file_info = bot.get_file(message.document.file_id)
            file_format = 'wav'
        case _:
            bot.delete_message(answer_message.chat.id, answer_message.message_id)
            bot.reply_to(message, f"Неподдерживаемый тип файла.")
            return
        
    print(f"Файл: {file_info.file_path}")
    downloaded_file = bot.download_file(file_info.file_path)
    try:
        phone_number, images = dtmf.recognize_dtmf(downloaded_file, file_format)
    except Exception:
        bot.edit_message_text("Ошибка при распознавании.", answer_message.chat.id, answer_message.message_id)
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
        
    bot.send_media_group(
        answer_message.chat.id,
        [telebot.types.InputMediaPhoto(img) for img in images]
    )



@bot.message_handler(func=lambda message: True)
def echo_message(message):
    global bot_mode
    if 'settings' in bot_mode:
        bot.reply_to(message, 'Нажми на кнопку параметра чтобы его выбрать')
    else:
        bot.reply_to(message, 'Я тебя не понимаю. Напиши /help.')



bot.infinity_polling()


# venv_dtmf\Scripts\activate
# python app\dtmf_tg_bot.py