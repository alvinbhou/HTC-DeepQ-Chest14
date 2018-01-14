# !/usr/bin/python 
# -*-coding:utf-8 -*- 
import sys, os
import time, datetime, random
import re, requests, json
import asyncio
import telepot
from telepot.aio.loop import MessageLoop
from telepot.aio.delegate import per_chat_id, create_open, pave_event_space, include_callback_query_chat_id
from telepot.namedtuple import InlineQueryResultArticle, InputTextMessageContent
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class User:
    def __init__(self, chatid):        
        self.chat_id = chatid

def getUser(chat_id):
    for user in users:
        if user.chat_id == chat_id:
            return user
    return None

def formatMsg(msg, score):
    score = random.randint(1,5)
    info = '若1為負面，5為正面\n'
    reply =  info + '『' + msg + '』可以獲得 ' + str(score) + '分'
    return reply

users = [] 
service_keyboard = ReplyKeyboardMarkup(
                            keyboard=[
                                [KeyboardButton(text="Feeling lucky!"),KeyboardButton(text="Help")], 
                            ]
                        )  


class IRBot(telepot.aio.helper.ChatHandler):

    def __init__(self, *args, **kwargs):
        super(IRBot, self).__init__(*args, **kwargs)

    async def on_chat_message(self, msg):      
        # two lines of keyboard
        keyboard1 = []                         
        keyboard2 = []   
        
        content_type, chat_type, chat_id = telepot.glance(msg)
        if(getUser(chat_id) is None):
            print("new user", chat_id)
            user = User(chat_id)
            users.append(user)

        msg = msg['text']
        print(chat_id, msg) 


        if msg == '/start':
            await self.sender.sendMessage( "您好！請隨意輸入字句我們會進行預測 :)", reply_markup=service_keyboard)
        elif msg == 'Help' or msg == '/help':
            await self.sender.sendMessage( "此為第17組的IR final Project", reply_markup=service_keyboard)
        elif msg == 'Feeling lucky!':
            myDictionary = ["早安我的朋友", "頂著幹，做中學，玩真的！",  "我大資管"]
            await self.sender.sendMessage( formatMsg(random.choice(myDictionary), 1), reply_markup=service_keyboard)
        else:
            # # # # # # # # # # # # # # # #
            #      KERAS MODEL HERE       #
            # # # # # # # # # # # # # # # #
            await self.sender.sendMessage( formatMsg(msg, 1), reply_markup=service_keyboard)
            
       
        
  
       
        return

            


TOKEN = sys.argv[1]  # get token from command-line

bot = telepot.aio.DelegatorBot(TOKEN, [
    include_callback_query_chat_id(
        pave_event_space())(
        per_chat_id(), create_open, IRBot, timeout= 120),
])

loop = asyncio.get_event_loop()
loop.create_task(MessageLoop(bot).run_forever())
print('Listening ...')
loop.run_forever()