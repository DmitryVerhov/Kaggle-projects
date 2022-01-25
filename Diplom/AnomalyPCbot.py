from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher.filters import Text
from aiogram.utils.markdown import hlink
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ParseMode
from aiogram.utils import executor

import pandas as pd
import os
from rating_predictor import *

'''This function finds the closest number to the given value'''
def nearest_value(items, value):
    found = items[0] 
    for item in items:
        if abs(item - value) < abs(found - value):
            found = item
    return found

# Loading NLP model
model = RatingClassifier()
model.load_model('oversample.bin')
sentiments =['Отрицательное','Нейтральное','Положительное']

# Loading bot
bot = Bot(token = os.getenv("TOKEN"), parse_mode = types.ParseMode.HTML)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Finite state machine:
class Sentiment(StatesGroup):
    predict = State()  
    save_result = State()  

class RAM(StatesGroup):
    form = State()
    type = State() 
    volume = State()
    n_planks = State()
    finish = State()

class NOTE(StatesGroup):
    start = State()
    finish = State()
    
# Startup  
async def on_startup(_):
    print('Token is valid, starting...')

#---------------------------------MAIN MEMU---------------------------------
@dp.message_handler(commands = 'start')
async def start(message: types.Message):
    start_buttons = ['Комплектующие','Ноутбуки','Анализ отношения']
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
    keyboard.add(*start_buttons)

    await message.answer('Выберите категорию', reply_markup = keyboard)

@dp.message_handler(Text(equals='Комплектующие'))
async def get_accesssories(message: types.Message):
    #accessory_buttons = ['Корпус','Охлаждение','HDD','МатПлаты',
    #                     'БП','Процессоры','RAM','SSD','Видеокарты']
    accessory_buttons = ['RAM']
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
    keyboard.add(*accessory_buttons)
    
    await message.answer('Выберите раздел', reply_markup = keyboard)

#---------------------------------RAM----------------------------------------
@dp.message_handler(Text(equals='RAM'),state=None)
async def ram_start(message: types.Message):
    await RAM.form.set()
    accessory_buttons = ['DIMM','SODIMM']
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
    keyboard.add(*accessory_buttons)
    
    await message.answer('Выберите форм-фактор',reply_markup = keyboard)

@dp.message_handler(state=RAM.form)
async def ram_form(message: types.Message, state: FSMContext):    
   
   async with state.proxy() as data:
       if message.text == 'DIMM':
            data['form'] = 1
       else:
           data['form'] = 0     
   
   accessory_buttons = ['ddr3','ddr4']
   keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True,
                                        one_time_keyboard=True)
   keyboard.add(*accessory_buttons)
   
   await RAM.next()
   await message.reply('Выберите тип',reply_markup = keyboard)
   
@dp.message_handler(state=RAM.type)
async def ram_type(message: types.Message, state: FSMContext):    
   async with state.proxy() as data:
       if message.text == 'ddr3':
        data['type'] = 1
       else:
        data['type'] = 0   
   await RAM.next()
   await message.reply('Какой объём?')

# Check volume, it must be digit
@dp.message_handler(lambda message: not message.text.isdigit(), state=RAM.volume)
async def process_volume_invalid(message: types.Message):
 
    return await message.reply("Нужно ввести целое число")

@dp.message_handler(lambda message: message.text.isdigit(),state=RAM.volume)
async def ram_volume(message: types.Message, state: FSMContext):    
   async with state.proxy() as data:
       data['volume'] = int(message.text)
   
   accessory_buttons = ['1','2','4']
   keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True,
                                        one_time_keyboard=True)
   keyboard.add(*accessory_buttons)
   
   await RAM.next()
   await message.reply('Сколько планок?',reply_markup = keyboard)

@dp.message_handler(state=RAM.n_planks)
async def ram_planks(message: types.Message, state: FSMContext):    
    async with state.proxy() as data:
       data['n_planks'] = int(message.text)
       data['counter'] = 0
    df = pd.read_csv('data/bot_data/ram.csv')
    vol = nearest_value(df.volume.unique(),data['volume'])
   
    try:
        df = df[(df.dimm == data['form'])&
                (df.ddr3 == data['type'])& 
                (df.volume == vol)& 
                (df.n_planks == data['n_planks'])].sort_values(['probabilities','rating','price'])
        
        begin = 0
        end = 5
        for i in range(begin,end):
                link = (f'{hlink(df.name.iloc[i],df.link.iloc[i])}')
                await message.answer(link)
        

        await RAM.next()
        
        accessory_buttons = ['Ещё пять','Хватит']
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
        keyboard.add(*accessory_buttons)

        await message.answer('Ещё?', reply_markup = keyboard)
    
    except:
        await state.finish()
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
        keyboard.add('/start')

        await message.answer('Ничего не найдено :-(', reply_markup = keyboard)

@dp.message_handler(state=RAM.finish)
async def ram_finish(message: types.Message, state: FSMContext):    
   if message.text == 'Хватит':
       await state.finish()
       keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
       keyboard.add('/start')
       await message.answer("Вернуться в меню?",reply_markup = keyboard)
   
   else:
        try:
            async with state.proxy() as data:
                data['counter'] += 5
            df = pd.read_csv('data/bot_data/ram.csv')
            vol = nearest_value(df.volume.unique(),data['volume'])
            
            df = df[(df.dimm == data['form'])&
                    (df.ddr3 == data['type'])& 
                    (df.volume == vol)& 
                    (df.n_planks == data['n_planks'])].sort_values(['probabilities','rating','price'])
            
            begin = data['counter']
            end = data['counter'] + 5
            for i in range(begin,end):
                    link = (f'{hlink(df.name.iloc[i],df.link.iloc[i])}')
                    await message.answer(link)
            

            accessory_buttons = ['Ещё пять','Хватит']
            keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True,
                                                    one_time_keyboard=True)
            keyboard.add(*accessory_buttons)

            await message.answer('Ещё?', reply_markup = keyboard)  
        
        except:
            await state.finish()
            keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True,
                                                    one_time_keyboard=True)
            keyboard.add('/start')

            await message.answer('Больше ничего не найдено :-(', reply_markup = keyboard)


#---------------------------------NOTEBOOKS---------------------------------
@dp.message_handler(Text(equals='Ноутбуки'),state=None)
async def note_start(message: types.Message):
    await NOTE.start.set()
    await message.answer('Введите ориентировочную цену')

@dp.message_handler(lambda message: not message.text.isdigit(), state=NOTE.start)
async def process_price_invalid(message: types.Message):
 
    return await message.reply("Нужно ввести целое число")

@dp.message_handler(lambda message: message.text.isdigit(),state=NOTE.start)
async def note_price(message: types.Message, state: FSMContext):    
    async with state.proxy() as data:
       data['price'] = int(message.text)
       data['counter'] = 0 
   
    df = pd.read_csv('data/bot_data/notebooks.csv')
       
    try:
        df = df[df.price.between(data['price']-5000,data['price']+5000)]\
                .sort_values(['probabilities','rating','price'])
        
        begin = 0
        end = 5
        for i in range(begin,end):
                link = (f'{hlink(df.name.iloc[i],df.link.iloc[i])}')
                await message.answer(link)
        
        await NOTE.next()
        
        accessory_buttons = ['Ещё пять','Хватит']
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
        keyboard.add(*accessory_buttons)

        await message.answer('Ещё?', reply_markup = keyboard)
    
    except:
        await state.finish()
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
        keyboard.add('/start')

        await message.answer('Ничего не найдено :-(', reply_markup = keyboard)

@dp.message_handler(state=NOTE.finish)
async def note_finish(message: types.Message, state: FSMContext):    
   if message.text != 'Ещё пять':
       await state.finish()
       keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
       keyboard.add('/start')
       await message.answer("Вернуться в меню?",reply_markup = keyboard)
   
   else:
        try:
            async with state.proxy() as data:
                data['counter'] += 5
            df = pd.read_csv('data/bot_data/notebooks.csv')
                       
            df = df[df.price.between(data['price']-5000,data['price']+5000)] \
                    .sort_values(['probabilities','rating','price'])
        
            
            begin = data['counter']
            end = data['counter'] + 5
            for i in range(begin,end):
                    link = (f'{hlink(df.name.iloc[i],df.link.iloc[i])}')
                    await message.answer(link)
            

            accessory_buttons = ['Ещё пять','Хватит']
            keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True,
                                                    one_time_keyboard=True)
            keyboard.add(*accessory_buttons)

            await message.answer('Ещё?', reply_markup = keyboard)  
        
        except:
            await state.finish()
            keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True,
                                                    one_time_keyboard=True)
            keyboard.add('/start')

            await message.answer('Больше ничего не найдено :-(', reply_markup = keyboard)    


#---------------------------------SENTIMENT ANALYSIS---------------------------------
@dp.message_handler(Text(equals='Анализ отношения'),state=None)
async def sentiment_start(message: types.Message):
    await Sentiment.predict.set()
    await message.answer('Введите текст')
    
@dp.message_handler(state=Sentiment.predict)
async def sentiment_predict(message: types.Message, state: FSMContext):
    
    async with state.proxy() as data:
       data['text'] = message.text
       data['predict'] = model.predict(message.text)
    last_list = sentiments.copy()
    del last_list[data['predict']]
    accessory_buttons = ['Да',f'Нет,{last_list[0]}',f'Нет,{last_list[1]}']
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
    keyboard.add(*accessory_buttons)
    prediction = sentiments[data['predict']]
    await Sentiment.next()
    await message.reply(f'{prediction}, верно?',reply_markup = keyboard)

@dp.message_handler(state=Sentiment.save_result)    
async def sentiment_save(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
       data['true'] = message.text
    
    '''Here will be the code to save results into the database'''
    
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard = True)
    keyboard.add('/start')
    if message.text == 'Да':
        await message.answer('Рад стараться :-)',reply_markup = keyboard)
    else:
        await message.answer('Буду тренироваться :-(',reply_markup = keyboard)
    await state.finish()

def main():
    executor.start_polling(dp,skip_updates =True,on_startup =on_startup)

if __name__ == "__main__":
    main()    
