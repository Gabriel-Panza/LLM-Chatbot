from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

print("Chatbot iniciado. Digite 'exit'/'quit'/'' para sair.\n")

while True:
    input_text = input("You: ")
    if input_text.strip().lower() in ['exit', 'quit', '']:
        print("Saindo do chatbot.")
        break

    conversation_history.append(f"<user>: {input_text}")

    # Limita o histórico para as últimas 8 interações
    max_turns = 8
    trimmed_history = conversation_history[-max_turns:]
    history_string = '\n'.join(trimmed_history)

    inputs = tokenizer(history_string, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=256, do_sample=True, top_k=50, temperature=0.7)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Bot: {response}")
    conversation_history.append(f"<bot>: {response}")