import telebot
from decouple import config
import re

from dtmf import DTMF


API_TOKEN = config('BOT_TOKEN')
bot = telebot.TeleBot(API_TOKEN)
dtmf = DTMF()


@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, (
        f'Привет, {message.from_user.first_name}!\n\n'
        'Я - бот для генерации и распознавания DTMF сигнала. 🤖\n\n'
        '📞 Для генерации DTMF сигнала отправьте мне номер телефона. Допустимые символы: 0-9, *, # и A-D.\n\n'
        '🎙️ Для распознавания DTMF сигнала отправьте мне голосовое сообщение.\n\n'
        '📚 Отправьте /info, чтобы узнать больше о DTMF и как его использовать.'
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


@bot.message_handler(content_types=['text'])
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


@bot.message_handler(content_types=['audio', 'voice', 'document'])
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

    downloaded_file = bot.download_file(file_info.file_path)

    phone_number, images = dtmf.recognize_dtmf(downloaded_file, file_format)

    if phone_number:
        bot.edit_message_text(
            f"Набран номер: {phone_number}",
            answer_message.chat.id,
            answer_message.message_id,
            disable_web_page_preview = True
        )
        bot.send_media_group(
            answer_message.chat.id,
            [telebot.types.InputMediaPhoto(img) for img in images]
        )
    else:
        bot.edit_message_text("Не удалось распознать номер.", answer_message.chat.id, answer_message.message_id)


@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.reply_to(message, 'Я тебя не понимаю. Напиши /help.')


bot.infinity_polling()


# venv_dtmf\Scripts\activate
# python app\dtmf_tg_bot.py