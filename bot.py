import logging
import requests
import transformation
import cyclegan

from aiogram import Bot, Dispatcher, executor, types

API_TOKEN = 'your token here'

token=API_TOKEN
url = "https://api.telegram.org/bot" + API_TOKEN + "/"

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def help_needed(message: types.Message):
    await message.answer("Привет! \n"
                         "Я - бот, который может перенести стиль одной картинки на другую. "
                         "Хочешь оформить свою фотографию так, как будто её рисовал Ван Гог? Пикассо? Твой друг Вася? "
                         "Тогда ты по адресу. "
                         "Вот смотри, для тебя, как для пользователя, я буду хранить и обрабатывать три картинки: content, style и vangogh. "
                         "А теперь поподробнее: \n"
                         "У меня есть два режима работы: \n"
                         "1. StyleTransfer - в этом режиме я могу сделать картинку взяв стиль с одной картинки и наложив на другую. "
                         "Отправь мне картинку с подписью 'style' (как ты уже понял, стиль мы возьмём с неё) "
                         "и другую картинку с подписью 'content' (именно её мы и будем перерисовывать). Потом отправь '/transform', немного подожди и "
                         "наслаждайся результатом!\n"
                         "2. CycleGAN - не парься, это просто название нейросети. В этом режиме я могу превратить любую твою картинку в картину Ван Гога. "
                         "Сеть была написана моим создателем, поэтому результаты далеки от идеальных в силу слабого железа. Чтобы опробовать этот режим - "
                         "отправь мне картинку с подписью 'vangogh' (преобразуем мы именно её). Потом отправь команду '/gan' и подожди некоторое время.\n\n"
                         "P.S.: Даже ботам иногда нужен отдых, иначе восстание машин неминуемо. Я работаю "
                         "на собственном сервере моего создателя, поэтому иногда мне нужен перерыв. Но с 8:00 МСК "
                         "до 20:00 МСК я всегда готов вам помочь!")
    print("some help needed")
    vangogh = open("examples/vangogh.jpg", "rb")
    style1 = open("examples/stylet1.jpg","rb")
    style2 = open("examples/stylet2.jpg","rb")
    await message.answer_photo(vangogh,"Пример того, как я работаю. Вот это - фото в стиле эскизных набросоков Ван Гога")
    await message.answer_photo(style1, "А это - перенос стиля: в данном случае - Ван Гога")
    await message.answer_photo(style2, "А здесь - перенос стиля потрясающей художницы из Питера, Зелёной Лампочки")



@dp.message_handler(content_types=['photo'])
async def photo_given(message: types.Message):
    photo_index = message.photo[0].file_id
    get_path = requests.get(url + "getFile?file_id=" + photo_index).json()['result']['file_path']
    picture_path = "https://api.telegram.org/file/bot"+token+"/" + get_path
    caption = message.caption.lower()
    if caption in ['content', 'style', 'vangogh']:
        with open(caption + "/" + str(message.chat.id) + '.jpg', 'wb') as handle:
            response = requests.get(picture_path, stream=True)
            for block in response.iter_content(1024):
                if not response.ok:
                    print(response)
                handle.write(block)
        if caption == 'vangogh':
            await message.answer('Прекрасно! Если хочешь преобразовать эту картинку - '
                                 'отправь /gan. \n'
                                 'Если хочешь поменять картинку - отправь новую снова с подписью "vangogh", '
                                 'преобразование будет проходить с последней картинкой, присланной с этой меткой.\n'
                                 'А ещё, я сделаю картинку такого размера, какого ты захочешь. Для этого тебе нужно после '
                                 '"/gan" указать размер желаемой картинки, то есть сообщение будет выглядеть как '
                                 '"/gan 500", число - это кол-во пикселей, ширина картинки. Если отправишь просто "/transform", то '
                                 'размер будет взят по умолчанию.\n'
                                 'При экспериментах учти, что чем большего размера картинку ты хочешь получить - тем дольше '
                                 'тебе придётся ждать. Так, картинка в 256 пикселей расчитывается где-то 0.5 мин, а в 1024 - уже целых 3.'
                                 'Ну что, начинаем? (ну давай, отправь уже "/gan")')
        else:
            try:
                content_img = transformation.Image.open("content/" + str(message.chat.id) + '.jpg')
            except:
                content_img = 0
                await message.answer('Прекрасно! Мне не хватает только картинки контента, отправь её с подписью "content"')
            try:
                style_img = transformation.Image.open("style/" + str(message.chat.id) + '.jpg')
            except:
                style_img = 0
                await message.answer('Прекрасно! Мне не хватает только картинки стиля, отправь её с подписью "style"')

            if style_img and content_img:
                await message.answer('Итак, обе картинки получены. Если хочешь, чтобы я начал генерацию новой картинки - '
                                     'отправь "/transform" отдельным соощением.\n'
                                     ' Если хочешь поменять стиль или контент - просто отправь другие картинки с '
                                     'надписью "content" либо "style", генерация будет происходить с последней '
                                     'из каждой категории после того, как ты запустишь процесс трансформации. '
                                     'А ещё, я сделаю картинку такого размера, какого ты захочешь. Для этого тебе нужно после \n'
                                     '"/transform" указать размер желаемой картинки, то есть сообщение будет выглядеть как '
                                     '"/transform 500", число - это кол-во пикселей, ширина картинки. Если отправишь просто "/transform", то '
                                     'размер будет взят по умолчанию.\n'
                                     'При экспериментах учти, что чем большего размера картинку ты хочешь получить - тем дольше '
                                     'тебе придётся ждать. Так, картинка в 256 пикселей расчитывается где-то 3 мин, а в 512 - уже целых 12.'
                                     'Ну что, начинаем? (ну давай, отправь уже "/transform")')
    else:
        await message.answer('Ой, в картинка должна быть подписана как "content", "style" или "vangogh"')
    print("photo was given")


@dp.message_handler(content_types=['document'])
async def photo_given(message: types.Message):
    photo_index = message.document.file_id
    get_path = requests.get(url + "getFile?file_id=" + photo_index).json()['result']['file_path']
    picture_path = "https://api.telegram.org/file/bot"+token+"/" + get_path
    caption = message.caption.lower()
    if caption in ['content', 'style', 'vangogh']:
        with open(caption + "/" + str(message.chat.id) + '.jpg', 'wb') as handle:
            response = requests.get(picture_path, stream=True)
            for block in response.iter_content(1024):
                if not response.ok:
                    print(response)
                handle.write(block)
        if caption == 'vangogh':
            await message.answer('Прекрасно! Если хочешь преобразовать эту картинку - '
                                 'отправь /gan. \n'
                                 'Если хочешь поменять картинку - отправь новую снова с подписью "vangogh", '
                                 'преобразование будет проходить с последней картинкой, присланной с этой меткой.\n'
                                 'А ещё, я сделаю картинку такого размера, какого ты захочешь. Для этого тебе нужно после \n'
                                 '"/gan" указать размер желаемой картинки, то есть сообщение будет выглядеть как '
                                 '"/gan 500", число - это кол-во пикселей, ширина картинки. Если отправишь просто "/transform", то '
                                 'размер будет взят по умолчанию.\n'
                                 'При экспериментах учти, что чем большего размера картинку ты хочешь получить - тем дольше '
                                 'тебе придётся ждать. Так, картинка в 256 пикселей расчитывается где-то 0.5 мин, а в 1024 - уже целых 3.'
                                 'Ну что, начинаем? (ну давай, отправь уже "/gan")')
        else:
            try:
                content_img = transformation.Image.open("content/" + str(message.chat.id) + '.jpg')
            except:
                content_img = 0
                await message.answer('Прекрасно! Мне не хватает только картинки контента, отправь её с подписью "content"')
            try:
                style_img = transformation.Image.open("style/" + str(message.chat.id) + '.jpg')
            except:
                style_img = 0
                await message.answer('Прекрасно! Мне не хватает только картинки стиля, отправь её с подписью "style"')

            if style_img and content_img:
                await message.answer('Итак, обе картинки получены. Если хочешь, чтобы я начал генерацию новой картинки - '
                                     'отправь "/transform" отдельным соощением.\n'
                                     ' Если хочешь поменять стиль или контент - просто отправь другие картинки с '
                                     'надписью "content" либо "style", генерация будет происходить с последней '
                                     'из каждой категории после того, как ты запустишь процесс трансформации. '
                                     'А ещё, я сделаю картинку такого размера, какого ты захочешь. Для этого тебе нужно после '
                                     '"/transform" указать размер желаемой картинки, то есть сообщение будет выглядеть как '
                                     '"/transform 500", число - это кол-во пикселей, ширина картинки. Если отправишь просто "/transform", то '
                                     'размер будет взят по умолчанию.\n'
                                     'При экспериментах учти, что чем большего размера картинку ты хочешь получить - тем дольше '
                                     'тебе придётся ждать. Так, картинка в 256 пикселей расчитывается где-то 3 мин, а в 512 - уже целых 12.'
                                     'Ну что, начинаем? (ну давай, отправь уже "/transform")')
    else:
        await message.answer('Ой, в картинка должна быть подписана как "content", "style" или "vangogh"')
    print("photo was given")


@dp.message_handler(commands=["transform"])
async def transformation_ask(message: types.Message):
    await message.answer(
        "Итак, я пока преобразую картинку, а ты на это время отодвинь телефон/компьютер и сделай зарядку!")
    try:
        imsize = int(message.text.split()[1])
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


@dp.message_handler(commands=["gan"])
async def gan_ask(message: types.Message):
    await message.answer(
        "Итак, я пока преобразую картинку, а ты на это время отодвинь телефон/компьютер и сделай зарядку!")
    try:
        imsize = int(message.text.split()[1])
    except:
        imsize = 1024
    gan = cyclegan.Gan("vangogh/" + str(message.chat.id) + '.jpg', "gan/" + str(message.chat.id) + ".jpg", imsize)
    ganed = open(gan.paint(), "rb")
    await message.answer_photo(ganed, "Если бы это фото было нарисовано Ван Гогом...")
    ganed.close()
    print("ganed")


@dp.message_handler()
async def extra_case(message: types.Message):
    await message.answer("Ой, что-то не так, я не знаю такой команды. \nЕсли нужна помощь - вызови '/help'")
    print("strange things happen")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
