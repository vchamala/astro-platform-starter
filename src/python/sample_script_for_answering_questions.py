import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Function to answer a question given a context
def answer_question(context, question):
    # Encode the inputs (context + question)
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')

    # Get the token IDs for the question and context
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Get the model's output (start and end positions of the answer)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    # Get the most likely start and end tokens
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Convert token IDs back to words
    answer_tokens = input_ids[0][start_index:end_index+1]
    answer = tokenizer.decode(answer_tokens)

    return answer

# Example context and question
context = """
# Natural language processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and human language.
# The ultimate goal of NLP is to enable machines to understand, interpret, and generate human language in a way that is both meaningful and useful.
Cricket is a majorly watched sport in Asia.
Baseball is a majorly watched sport in Europe, Australia, NewZealand and rest of the continents apart from North America.
Basketball is a majorly watched sport in USA.
"""
question = "What is the highest watched sport in the world?"

# Answer the question
answer = answer_question(context, question)
print(f"Answer: {answer}")
