from transformers import T5Tokenizer, TFT5ForConditionalGeneration, T5ForConditionalGeneration
# the model will be downloaded automatically from Huggingface's model hub
model_name_or_path = "congcongwang/t5-large-fine-tuned-wnut-2020-task3"
# Tensorflow2.0
model = TFT5ForConditionalGeneration.from_pretrained(model_name_or_path)

# or PyTorch
# model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)

source = "context: *Prince Charles tests positive for Corona* Prince William knowing he's " \
         "the next in line to the throne: https://t.co/B1nmIpLj69. question: Who is tested positive?" \
         "choices: author of the tweet, not specified, the next in line, line to the throne, *Prince Charles," \
         " Corona* Prince William, he, the next, line, the throne."

inputs = tokenizer.encode(source, return_tensors="tf")  # Batch size 1. change "tf" to "pt" if using pytorch model
result = model.generate(inputs)

print(tokenizer.decode(result[0]))
# output: Prince Charles
