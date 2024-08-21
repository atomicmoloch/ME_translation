import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
infer = AutoModelForSeq2SeqLM.from_pretrained("Qilex/bart-base-me-en")

def translate(sentence):
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    outputs = infer.generate(input_ids, max_new_tokens = len(sentence.split(' '))*10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_multiline(sentence):
#    if len(sentence.split()) > 300:
#        print('Please insert less text')
    if '\n' in sentence:
        lines = sentence.split('\n')
        translated_lines = [translate(line) for line in lines if len(line) > 0]
        return '\n'.join(translated_lines)
    else:
        return translate(sentence)

title = "Modern English to Middle English Translator"
description = """
This translator is trained on about 70,000 English/Middle English paired sentences.
<br>
It's still a work in progress.
<br>
"""
article = '''
<br>
You can improve results by removing contractions (hadn't -> had not)
'''

gr.Interface(
    fn=translate_multiline,
    inputs=gr.Textbox(lines=1, placeholder="Enter text to translate."),
    outputs="text",
    title=title,
    description=description,
    article = article,
).launch()
