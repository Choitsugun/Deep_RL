from hyperparams import set_args
from inference import Inference
from tqdm import tqdm
from utils import*
import os
import uvicorn
from fastapi import FastAPI
from collections import defaultdict

class Chatbot():
    model = None
    def load_model(self, actor_name, method):
        args = set_args()
        logger = create_logger(args)

        #device_assign(args, logger)
        #device_assignが２回実行できない為、args.deviceだけ抽出して実行
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.actor_name = actor_name
        args.Forinte_checkp = self.get_forinte_checkp(actor_name, method)

        #print("loading model...")
        #print("actor_name:", args.actor_name)
        #print("Forinte_checkp:", args.Forinte_checkp)
        self.model = Inference(args, logger, if_interact=True).to(args.device)

    def get_forinte_checkp(self, actor_name, method):
        if method == "MLE":
            return f"../save_load/model/actor/wo_tp/{actor_name}/epoch20"
        elif method == "RL":
            return f"../save_load/model/actor_im/stan/{actor_name}/epoch20"
        elif method == "ours":
            return f"../save_load/model/actor_im/ours/{actor_name}/epoch20"
        else:
            #print(f"method:{method} is not found.")
            return None

    def generate(self, message_list):
        response = self.model.generate(message_list)[0]
        return response.lstrip()

app = FastAPI()
actor_dict = defaultdict(Chatbot)


@app.post("/generate") # POSTに書き換える。(?で分割)
def generate(data: dict):
    """
    data : {
        dialog: str
        actor_name: str
        method: str
        }

    dialog examples : 
    - sentence1
    - sentence1¥tsentence2
    - sentence1¥tsentence2¥tsentence3
    """
    
    dialog = data["dialog"]
    message_list = dialog.split("¥t")
    key = f"{data['actor_name']}_{data['method']}"
    #print("key:", key)

    # load model (only initial use)
    #if actor.model == None:
    #    actor.load_model(data["actor_name"], data["method"])
    #response = actor.generate(message_list)
    
    # load model (only initial use)
    if actor_dict[key].model == None:
        actor_dict[key].load_model(data["actor_name"], data["method"])
    response = actor_dict[key].generate(message_list)
    return response


if __name__ == "__main__":
    args = set_args()
    logger = create_logger(args)
    device_assign(args, logger)
    uvicorn.run("chatbot_api:app", host="0.0.0.0", port=5050)



    
