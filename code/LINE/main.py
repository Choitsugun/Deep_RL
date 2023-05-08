# 必要モジュールの読み込み
from flask import Flask, request, abort
import requests
import os
from pathlib import Path
import json
import datetime
import random
from collections import defaultdict
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

# 変数appにFlaskを代入。インスタンス化
app = Flask(__name__)


#環境変数取得
# chatbot
CHANNEL_ACCESS_TOKEN = "a/T6i9+0Q06F+MqrWgKC8vKDBtX8tRqcWewZATLY2Fs+DEUP+DhofGau9FjLCyGEKbC+Tphwa0/51a8plqc1HFfmQmRffk4+M40TXo7XEkz+emlXkSdov9qkTLrXC65/ZWZ2CG6EqCLhwQE6fjAp/Y9PbdgDzCFqoOLOYbqAITQ="
CHANNEL_SECRET = "772b552a6b12f701299d5ed1d928e421"

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)


# 設定
actor_names = ["GPT2", "DialoGPT", "T5", "GODEL"]
methods = ["MLE", "RL", "ours"]
eval_list = ["Quality", "Informativeness", "Empathy","Engagingness"]
turn_threshold = 5  # 最低限会話しないといけないターン数
context_length = 3  # 考慮するコンテキストの長さ
mail_adress = "choitsugun@keio.jp"
dummy_message = "Let's change the topic and talk about something else." #空白とスペースが返答された時に変換するダミーメッセージ


class User:
    def __init__(self):
        self.mode = "sleep"
        self.dialog = []
        self.exp_id = 0
        self.eval_id = 0
        self.evaluation = []
        self.is_debug = False
        self.is_confident = False
        self.exp_list = [f"{actor}_{method}" for actor in actor_names for method in methods]
        self.shuffle_exp_list()
        self.set_model()
    
    def set_model(self):
        actor, method = self.exp_list[self.exp_id].split("_")
        self.actor_name = actor
        self.method = method
    
    def shuffle_exp_list(self):
        random.shuffle(self.exp_list)

    def reset(self):
        self.mode = "sleep"
        self.dialog = []
        self.exp_id = 0
        self.eval_id = 0
        self.evaluation = []
        self.set_model()

# ユーザー辞書の作成
user_dict = defaultdict(User)

# ユーザーからメッセージが送信された際、LINE Message APIからこちらのメソッドが呼び出される。
@app.route("/callback", methods=['POST'])
def callback():
    # リクエストヘッダーから署名検証のための値を取得
    signature = request.headers['X-Line-Signature']

    # リクエストボディを取得
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # 署名を検証し、問題なければhandleに定義されている関数を呼び出す。
    try:
        handler.handle(body, signature)
    # 署名検証で失敗した場合、例外を出す。
    except InvalidSignatureError:
        abort(400)
    # handleの処理を終えればOK
    return 'OK'

# LINEでMessageEvent（普通のメッセージを送信された場合）が起こった場合に、
# def以下の関数を実行します。
# reply_messageの第一引数のevent.reply_tokenは、イベントの応答に用いるトークンです。 
# 第二引数には、linebot.modelsに定義されている返信用のTextSendMessageオブジェクトを渡しています。
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event, source):
    user_message = event.message.text
    user_id = event.source.user_id

    # Userインスタンスが初めて生成された時に自動的にactor_nameとmethodを割り当てる。
    #if user_id not in user_dict.keys():
    #    user_dict[user_id].id = len(user_dict)
    #    print("add user:", user_dict[user_id].id)
    #    print("actor_name:", user_dict[user_id].actor_name)
    #    print("method:", user_dict[user_id].method)
   
    if user_message == "debug mode":
        if user_dict[user_id].is_debug == False:
            reply_message = "enter password"
            user_dict[user_id].is_debug = True
            line_bot_api.reply_message(
                event.reply_token, 
                [TextSendMessage(reply_message)])
        return

    if user_dict[user_id].is_debug:
        if not user_dict[user_id].is_confident:
            if user_message == "793110":
                user_dict[user_id].is_confident = True
                reply_message = "switched to debug mode\n[command list]\nend\nmodel info\nload all model\nreset\nspace\nempty\nlist\nnext"
            else:
                reply_message = "permission denied"
                user_dict[user_id].is_debug = False
            line_bot_api.reply_message(
                event.reply_token, 
                [TextSendMessage(reply_message)])
            return 

        if user_message == "end" or user_message == "'end'":
            user_dict[user_id].is_debug = False
            user_dict[user_id].is_confident = False
            reply_message = "finish debug mode"
        
        elif user_message == "model info":
            reply_message = f"actor_name: {user_dict[user_id].actor_name}\nmethod: {user_dict[user_id].method}"
       
        elif user_message == "reset":
            user_dict[user_id].reset()
            reply_message = "reset your data"
        elif user_message == "space":
            reply_message = " "
        elif user_message == "empty":
            reply_message = ""
        elif user_message == "load all model":
            for actor_name in actor_names:
                for method in methods:
                    user_dict[user_id].dialog = ["hi"]
                    user_dict[user_id].actor_name = actor_name
                    user_dict[user_id].method = method
                    generate_response(user_id)
            user_dict[user_id].reset()
            reply_message = "all models was loaded"
        elif user_message == "list":
            reply_message = ""
            for key in user_dict[user_id].exp_list:
                reply_message += key + "\n"
        elif user_message == "next":
            if user_dict[user_id].exp_id >= len(methods)*len(actor_names) - 1 :
                reply_message = "last"
            else:
                user_dict[user_id].exp_id += 1
                user_dict[user_id].set_model()
                reply_message = user_dict[user_id].actor_name + "_" + user_dict[user_id].method
        else:
            reply_message = "[command list]\nend\nmodel info\nload all model\nreset\nspace\nempty\nlist\nnext"

        line_bot_api.reply_message(
                event.reply_token, 
                [TextSendMessage(reply_message)])
        return 

    if user_dict[user_id].mode == "sleep":
        if user_message == "start" or user_message  == "'start'":
            user_dict[user_id].mode = "chat"
            reply_message = "Let's start the conversation.\nPlease send the first message with specific information, for example: 'I watched a sci-fi movie today, named The Wandering Earth 2.'"
        else:
            reply_message = f"Complete: {user_dict[user_id].exp_id}/{len(methods)*len(actor_names)}\nPlease type 'start' to begin the chat."
        line_bot_api.reply_message(
                event.reply_token, 
                [TextSendMessage(reply_message)])
        return

    elif user_dict[user_id].mode == "chat":
        # 会話の終了処理
        if user_message == "end" or user_message == "'end'":
            # 会話の終了条件の検証
            if len(user_dict[user_id].dialog) < turn_threshold*2:
                reply_message = "To end the conversation, please engage in a dialogue of 5 or more exchanges."
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=reply_message))
                return 

            # dialogの作成
            dialog = f"[Conversation log]\n"
            for i, message in enumerate(user_dict[user_id].dialog):
                name = "you" if i % 2 == 0 else "AI"
                dialog += f"{name}: {message}\n" # option + ¥
            dialog = dialog[:-1]

            # modeの変更
            user_dict[user_id].mode = "eval"
            description = "[Quality] measures the coherence and grammatical correctness of the responses.\n\
[Informativeness] measures the diversity and hallucination of the responses.\n\
[Empathy] measures the degree to which chatbot respond with concern or affectivity.\n\
[Engagingness] measures the desire to talk with the chatbot for a long conversation."
            line_bot_api.reply_message(
                    event.reply_token, 
                    [TextSendMessage(dialog),
                        TextSendMessage(f"Please rate the following criteria using 0, 1, or 2. The higher the score, the better.\n-----\n{description}"),
                        TextSendMessage("Please type 0, 1, or 2 to rate the Quality of the conversation.\n\
0: Most responses are incoherent or contain grammatical errors, preventing the dialogue from proceeding.\n\
1: Although some responses are incoherent or contain grammatical errors, the dialogue can continue.\n\
2: Only a few (or no) incoherent or grammatical errors in the responses, and the overall dialogue flows fluently.")])

            return

        # ユーザーからのメッセージをdialogに追加
        user_dict[user_id].dialog.append(user_message)
        # generate response
        generate_response(user_id)

        if len(user_dict[user_id].dialog) >= turn_threshold*2:
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text=user_dict[user_id].dialog[-1]), TextSendMessage(text="(※If you want to end the conversation and proceed to rating, please type 'end'. Otherwise, you may continue the conversation as long as you like.)")])
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=user_dict[user_id].dialog[-1]))
        return

    elif user_dict[user_id].mode == "eval":
        if user_message not in ["0", "1", "2"]:
            reply_message = f"There seems to be an error in your input. Please rate the {eval_list[user_dict[user_id].eval_id]} by typing 0, 1, or 2"
            line_bot_api.reply_message(
                    event.reply_token, 
                    TextSendMessage(reply_message))
            return 

        user_dict[user_id].evaluation.append((eval_list[user_dict[user_id].eval_id], user_message))
        user_dict[user_id].eval_id += 1

        if user_dict[user_id].eval_id < len(eval_list):
            if eval_list[user_dict[user_id].eval_id] == "Informativeness":
                description = "0: Most responses simply repeat information from the context or are generic.\n\
1: The information conflicts with common sense or contradicts the previous statement.\n\
2: Most responses have the appropriate information."

            if eval_list[user_dict[user_id].eval_id] == "Empathy":
                description = "0: Most responses were short or showed little concern for the users in the dialogue.\n\
1: Although not very coherent, some responses convey an emotional tone or ask a question.\n\
2: Some responses are both coherent and show care for or emotional attachment to the user."

            if eval_list[user_dict[user_id].eval_id] == "Engagingness":
                description = "0: The responses are lackluster or of poor quality, which makes it challenging to sustain the dialogue.\n\
1: The responses are not particularly engaging, but they are adequate for continuing the dialogue.\n\
2: The responses are engaging and have the potential to develop the dialogue."

            reply_message = f"Please type 0, 1, or 2 to rate the {eval_list[user_dict[user_id].eval_id]} of the conversation.\n{description}"
            line_bot_api.reply_message(
                        event.reply_token, 
                        TextSendMessage(reply_message))
            return 

        # 実験番号の増加
        user_dict[user_id].exp_id += 1
        # 評価ログの作成
        eval_log = "[Evaluation log]\n"
        for item, score in user_dict[user_id].evaluation:
            eval_log += f"{item}: {score}\n"
        # save
        save_result(user_id)
        #初期化
        user_dict[user_id].dialog = list()
        user_dict[user_id].eval_id = 0
        user_dict[user_id].evaluation = list()

        if user_dict[user_id].exp_id < len(methods)*len(actor_names):
            #次の実験の準備    
            user_dict[user_id].mode ="sleep"
            user_dict[user_id].set_model()
            reply_message = f"{eval_log}\nComplete: {user_dict[user_id].exp_id}/{len(methods)*len(actor_names)}\nTo begin the next chat, please type 'start'."
        else:
            user_dict[user_id].mode ="end"
            reply_message = f"{eval_log}\nAll experiments have concluded.\nThank you for participating in all the experiments.\nYour cooperation is greatly appreciated.\nSincerely."

        line_bot_api.reply_message(
                event.reply_token, 
                [TextSendMessage(reply_message)])
        return

    if user_dict[user_id].mode == "end":
        reply_message = f"The experiment has ended. Thank you for your cooperation. If you have any questions, please contact the following email address:{mail_adress}"
        line_bot_api.reply_message(
                event.reply_token, 
                [TextSendMessage(reply_message)])


def generate_response(user_id):
    #gypsum
    url = "http://131.113.46.22:80/generate"
    headers = {"Content-Type": "application/json"}
    # APIに送信するデータを作成する
    input_text = "¥t".join(user_dict[user_id].dialog[-context_length:])
    data = {
            "dialog": input_text,
            "actor_name": user_dict[user_id].actor_name,
            "method": user_dict[user_id].method,
    }
    response = requests.post(url, headers=headers, json=data)
    data = response.json()
    
    # 空白もしくは空文字の時
    if data.strip() == "":
        data = dummy_message

    user_dict[user_id].dialog.append(data)
    return

def save_result(user_id):
    save_dir = Path(f"./result/{user_id}")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{user_dict[user_id].actor_name}_{user_dict[user_id].method}.txt"
    with open(os.path.join(save_dir, filename), 'w') as f:
        f.write(f"actor_name: {user_dict[user_id].actor_name}\n")
        f.write(f"method: {user_dict[user_id].method}\n")
        f.write("[dialog]\n")
        for message in user_dict[user_id].dialog:
            f.write(f"{message}\n")
        f.write("[evaluation]\n")
        for item, score in user_dict[user_id].evaluation:
            f.write(f"{item}: {score}\n")

    


# ポート番号の設定
if __name__ == "__main__":
#    app.run()
    port = 5000
    app.run(host="0.0.0.0", port=port)