import logging
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import configparser
import time
from DetectFake import MultiTransform, SingleTransform, OpenCVSingleTransform, SingleCTransform
import cv2
import datetime
import pdb

config = configparser.ConfigParser()
config.read('config.ini')


TOKEN = config['TELEGRAM']['TOKEN']

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("start - 打招呼\nhelp - 列出可用指令")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)




# 定義一個處理圖片的函數
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 從更新中獲取接收到的圖片
    photo = update.message.photo[-1]
    
    # 從 Bot API 獲取文件對象
    photo_file = await update.message.photo[-1].get_file()

    # 下載圖片
    fileName = 'Photo/photo.jpg'
    await photo_file.download_to_drive(fileName)

    # 回覆用戶收到圖片
    await update.message.reply_text('收到圖片，正在處理中！')
    
    # 讀取圖片
    image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)

    ref = 0
    if(image.shape == (512,512)):
        ref = 0.4
    else:
        ref = 1
    
    # 傳入辨識程式
    start = time.time()
    result = MultiTransform(image)
    # result = SingleTransform(image)
    # result = OpenCVSingleTransform(image)
    result = str('{:.3f}'.format(result))
    end = time.time()
    useTime = end - start
    useTime = str('{:.3f}'.format(useTime))
    await update.message.reply_text('處理時間: {}s\n修圖程度: {}\n數字越大代表修得越大，大於 {} 應該就是有修。'.format(useTime, result, ref))




def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
