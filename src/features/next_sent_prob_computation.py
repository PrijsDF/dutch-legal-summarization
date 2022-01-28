from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer

# Code from https://stackoverflow.com/questions/55111360/using-bert-for-next-sentence-prediction
# For some reason using two sentences gives less UNK tokens than using two lists of tokens...
tseq_A = ['Hallo', 'hoe', 'gaat', 'het']
tseq_B = ['Goed', 'hoor']

seq_A = 'Hallo hoe gaat het'
seq_B = 'Goed hoor'

# load pretrained model and a pretrained tokenizer
model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# encode the two sequences. Particularly, make clear that they must be
# encoded as "one" input to the model by using 'seq_B' as the 'text_pair'
encoded = tokenizer.encode_plus(tseq_A, text_pair=tseq_B, return_tensors='pt')
print(encoded)
for ids in encoded["input_ids"]:
    print(tokenizer.decode(ids))
# a model's output is a tuple, we only need the output tensor containing
# the relationships which is the first item in the tuple
seq_relationship_logits = model(**encoded)[0]

# we still need softmax to convert the logits into probabilities
# index 0: sequence B is a continuation of sequence A
# index 1: sequence B is a random sequence
probs = softmax(seq_relationship_logits, dim=1)

# print(seq_relationship_logits)
print(probs)

# tensor([[9.9993e-01, 6.7607e-05]], grad_fn=<SoftmaxBackward>)
# very high value for index 0: high probability of seq_B being a continuation of seq_A

print(f'The probability that sequence B follows sequence A is: {round(probs[0][0].item(), 4)}')