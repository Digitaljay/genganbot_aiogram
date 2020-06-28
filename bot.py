import logging
import requests
import transformation

from aiogram import Bot, Dispatcher, executor, types

API_TOKEN = '1233264025:AAHEMen7FR6yhRZiVv1gi91z3COoEmQAOHo'
url = "https://api.telegram.org/bot" + API_TOKEN + "/"

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start', 'help'])
async def help_needed(message: types.Message):
    await message.answer("Я - бот, который может перенести стиль одной картинки на другую. "
                         "Хочешь офрмить свою фотографию так, как будто её рисовал Ван Гог? Пикассо? Твой друг Вася? "
                         "Тогда ты по адресу. Отправь мне картинку с подписью 'style'(как ты уже понял, стиль мы возьмём с неё) "
                         "и другую картинку с подписью 'content' (именно её мы и будем перерисовывать). Потом отправь '/transform' "
                         "и наслаждайся результатом!\n\n"
                         "P.S.: Даже ботам иногда нужен отдых, иначе восстание машин неминуемо. Я работаю"
                         "на собственном сервере моего создателя, поэтому иногда мне нужен перерыв. Но с 7:00 МСК "
                         "до 10:00 МСК я всегда готов вам помочь!")
    print("some help needed")


@dp.message_handler(content_types=['photo'])
async def photo_given(message: types.Message):
    photo_index = message.photo[0].file_id
    get_path = requests.get(url + "getFile?file_id=" + photo_index).json()['result']['file_path']
    picture_path = "https://api.telegram.org/file/bot1233264025:AAHEMen7FR6yhRZiVv1gi91z3COoEmQAOHo/" + get_path
    caption = message.caption.lower()
    if caption in ['content', 'style']:
        with open(caption + "/" + str(message.chat.id) + '.jpg', 'wb') as handle:
            response = requests.get(picture_path, stream=True)
            for block in response.iter_content(1024):
                if not response.ok:
                    print(response)
                handle.write(block)
        try:
            content_img = transformation.Image.open("content/" + str(message.chat.id) + '.jpg')
        except:
            content_img=0
            await message.answer( 'Прекрасно! Мне не хватает только картинки контента, отправь её с подписью "content"')
        try:
            style_img = transformation.Image.open("style/" + str(message.chat.id) + '.jpg')
        except:
            style_img=0
            await message.answer( 'Прекрасно! Мне не хватает только картинки стиля, отправь её с подписью "style"')

        if style_img and content_img:
            await message.answer( 'Итак, обе картинки получены. Если хочешь, чтобы я начал генерацию новой картинки - '
                                 'отправь "/transform" отдельным соощением.'
                                 ' Если хочешь поменять стиль или контент просто отправь другие картинки с '
                                 'надписью "content" либо "style", генерация будет происходить с последней '
                                 'из каждой категории после того, как ты запустишь процесс трансформации. '
                                 'А ещё, я сделаю картинку такого размера, какого ты захочешь. Для этого тебе нужно после '
                                  '"/transform" указать размер желаемой картинки, то есть сообщение будет выглядеть как '
                                  '"/transform 500", число - это кол-во пикселей. Если отправишь просто "/transform", то '
                                  'размер будет взят по умолчанию.\n'
                                  'При экспериментах учти, что чем большего размера картинку ты хочешь получить - тем дольше '
                                  'тебе придётся ждать. Так, картинка в 256 пикселей расчитывается где-то 3 мин, а в 512 - уже целых 12.'
                                 'Ну что, начинаем? (ну давай, отправь уже "/transform")')
    else:
        await message.answer('Ой, в картинка должна быть подписана как "content" или "style"')
    print("photo was given")


@dp.message_handler(commands=["transform"])
async def transformation_ask(message: types.Message):
    print("Итак, я пока преобразую картинку, а ты на это время отодвинь телефон/компьютер и сделай зарядку!")
    try:
        imsize=int(message.text.split()[1])
    except:
        imsize = 256
    content_img = "content/" + str(message.chat.id) + '.jpg'
    style_img = "style/" + str(message.chat.id) + '.jpg'
    transformator = transformation.Transfer(imsize, style_img, content_img)
    transformator.prepare_images()
    transformator.transform("results/" + str(message.chat.id) + ".jpg")
    photo = open("results/" + str(message.chat.id) + ".jpg", 'rb')
    await message.answer_photo(photo, "Та-да!")
    photo.close()
    print("transformation happened already!")


@dp.message_handler()
async def extra_case(message: types.Message):
    await message.answer("hey")
    print("strange things happen")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
