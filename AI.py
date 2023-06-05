from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers.utils import logging
logging.set_verbosity_error()


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

chat_history_ids = torch.Tensor()
step=0

while True:
    user_input = input("User: ")
    if "bye" in user_input.lower():
        print("DialoGPT: Goodbye!")
        break
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to('cpu')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids,
        do_sample=True, 
        max_length=1500,
        top_k=50, 
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        length_penalty=1.5,  # Adjust the length penalty
        num_return_sequences=1,  # Increase the number of returned sequences
        output_hidden_states=False,
        output_attentions=False,
        max_new_tokens=200,  # Increase the number of new tokens in the response
        temperature=0.7
    )

    # pretty print last ouput tokens from bot
    print("Chatbot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))) 
    
    step+=1


