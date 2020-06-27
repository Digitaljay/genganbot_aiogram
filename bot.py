import logging
import requests
import transformation

from aiogram import Bot, Dispatcher, executor, types

API_TOKEN = '1233264025:AAHEMen7FR6yhRZiVv1gi91z3COoEmQAOHo'
url = "https://api.telegram.org/bot" + API_TOKEN + "/"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

print("preparations are done now")


@dp.message_handler(commands=['start', 'help'])
async def echo(message: types.Message):
    await message.answer("help for u")
    print("some help needed")


@dp.message_handler(content_types=['photo'])
async def echo(message: types.Message):
    photo_index = message.photo[0].file_id
    get_path = requests.get(url + "getFile?file_id=" + photo_index).json()['result']['file_path']
    picture_path = "https://api.telegram.org/file/bot1233264025:AAHEMen7FR6yhRZiVv1gi91z3COoEmQAOHo/" + get_path
    caption = message.caption.lower()
    if caption in ['content', 'style']:
        with open(caption + str(message.chat.id) + '.jpg', 'wb') as handle:
            response = requests.get(picture_path, stream=True)
            for block in response.iter_content(1024):
                if not response.ok:
                    print(response)
                handle.write(block)
        await message.answer('Saved!')
    else:
        await message.answer('Wrong caption')
    print("photo was given")


@dp.message_handler(commands=["transform"])
async def echo(message: types.Message):
    print("trying to transform")
    await message.answer("Идёт преобразование, это займёт около трёх-четырёх минут, пока встань и сделай зарядку!")
    content_img = "content" + str(message.chat.id) + '.jpg'
    style_img = "style" + str(message.chat.id) + '.jpg'
    transformator = transformation.Transfer(256, style_img, content_img)
    transformator.prepare_images()
    transformator.transform("results/" + str(message.chat.id) + ".jpg")
    photo = open("results/" + str(message.chat.id) + ".jpg", 'rb')
    await message.answer_photo(photo, "Transformed specially for u!")
    photo.close()
    print("trnsformation happened already!")


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer("hey")
    print("strange things happen")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
