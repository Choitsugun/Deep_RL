from transformers import pipeline, Conversation
import random
import os

class BlenderChatBot():
    def __init__(self, args, logger):
        self.chatbot = pipeline(task="conversational", model=args.blender_model, device=int(os.environ["CUDA_VISIBLE_DEVICES"])) 
        self.logger = logger
        self.context = []
        self.dummy_actor_message = args.dummy_actor_mess
        self.num_state = args.blende_num_state
        with open(args.first_mess_path, "r") as f:
            self.first_messages = f.readlines()
            self.first_messages = [text.rstrip("\n") for text in self.first_messages]
        assert self.num_state % 2 == 1 # always odd
        
    def generate(self):
        context_block = self.make_context_block()
        conversation = self.chatbot(context_block)
        
        blender_response = conversation.generated_responses[-1]
        self.context.append(blender_response)
        
        return self.context
    
    def append_responce(self, responce: list):
        self.context.append(responce[0])
    
    def make_context_block(self):
        if len(self.context) > self.num_state:
            context = self.context[-self.num_state:]
        elif len(self.context) % 2 == 0:
            # add dummy message into context for multi-turn 
            context = [self.dummy_actor_message] + self.context
        else:
            context = self.context
        
        text = context[-1] # latest_user_input
        if len(context) >= self.num_state and self.num_state >= 3:
            generated_responses = list(reversed(context[-2::-2]))
            past_user_inputs = list(reversed(context[-3::-2]))
        else:
            generated_responses = None
            past_user_inputs = None
        
        context_block=Conversation(text = text, past_user_inputs =past_user_inputs, generated_responses=generated_responses)
        return context_block
    
    def reset_context(self):
        self.context = []
        
    def set_first_message(self):
        assert len(self.context) == 0
        self.context.append(random.choice(self.first_messages))

    def get_context(self):
        return self.context