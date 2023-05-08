from blender_chatbot import BlenderChatBot
from hyperparams import set_args
from inference import Inference
from tqdm import tqdm
from utils import*
import os

def main():
    device_assign(args, logger)
    
    # load model
    actor = Inference(args, logger, if_interact=True).to(args.device)
    blender = BlenderChatBot(args, logger)
    
    # make directory
    os.makedirs(args.save_inter_path, exist_ok=True)
    # remove old dialog
    save_inter_path = os.path.join(args.save_inter_path, "interaction.txt")
    if os.path.exists(save_inter_path):
        os.remove(save_inter_path)
    
    for i in tqdm(range(args.n_episode)):
        blender.reset_context()
        for j in range(args.n_turns):
            # get a responce of blender
            if j == 0:
                blender.set_first_message()
            else:
                blender.generate()
            # get a responce of actor
            context = blender.get_context()
            if len(context) > args.num_state:
                context = context[-args.num_state:]
            actor_responce = actor.generate(context)
            blender.append_responce(actor_responce)
            
        # save dialog
        with open(save_inter_path, "a") as f:
            for i, text in enumerate(blender.get_context()):
                text = text.strip()
                f.write(f"{text}\n")
            f.write("\n")


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    main()