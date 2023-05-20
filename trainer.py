# Python script to train CharGPT models.

from collections.abc import Callable
import torch, torch.nn as nn
torch.manual_seed(123)


        
def create_vocabulary(data: str, visualize: bool = False) -> (list[str], dict[str, int], dict[int, str], Callable[str, list[int]], Callable[list[int], str]):
    vocabulary = sorted(list(set(data)))
    token_to_index_map = {token:index for (index, token) in enumerate(vocabulary)}
    index_to_token_map = {index:token for (index, token) in enumerate(vocabulary)}

    if visualize:
        print(f'Visualizing vocabulary.')
        print(f'Length of vocabulary: {len(vocabulary)}.')
        print(f'Vocabulary is {"".join(vocabulary)}.')
        print(f'Token to index map sorted is {token_to_index_map}')
        print(f'Index to token map sorted is {index_to_token_map}')              
        
    def encoder(input: str) -> (list[int]):
        ''' Encodes the input string. 
            
            Args:
                input: string of text to be encoded.
                
            Returns:
                List of indices of the tokens in the input string.
        '''
        return [token_to_index_map[token] for token in input]
    
    def decoder(input: list[int]) -> str:
        ''' Decodes the input token index into text.
        
            Args:
                input: List of indices of tokens in the text to be decoded.
                
            Returns:
                String corresponding to the decoded text.
        '''
        return ''.join([index_to_token_map[index] for index in input])
        
    return (vocabulary, token_to_index_map, index_to_token_map, encoder, decoder)

def run_tokenizer_example(run: bool = False) -> None:
    ''' Run example text using character level tokenizer.'''
    if run:
        print('Running Tokenizer example.')
        input_text = 'Hello, how are you?'
        tokenized_text = encoder(input_text)
        decoded_text = decoder(tokenized_text)
        print(f'{input_text=}, {tokenized_text=}, {decoded_text=}.')
        
def visualize_batch(x, y, skip_visualization: bool = True):
    if not skip_visualization:
        for sample in range(x.shape[0]):
            for context in range(x.shape[1]):
                print(f' Context: {x[sample, :context+1]}. Target: {y[sample, context]}.')

def create_batch(split, block_size, batch_size):
    data = train_set if split == 'train' else val_set
    batch_start_index = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in batch_start_index])
    y = torch.stack([data[i+1: i+1+block_size] for i in batch_start_index])
    return (x,y)
            
@torch.no_grad()
def evaluate_loss(batch_index, model, batch_size, block_size):
    model.eval()

    (x, y) = create_batch('train', block_size, batch_size)
    predictions = model(x)
    (B, T, C) = predictions.shape
    predictions = predictions.view(B*T, C)
    y = y.view(-1)
    train_loss = nn.functional.cross_entropy(predictions, y)
    
    (x, y) = create_batch('val', block_size, batch_size)
    predictions = model(x)
    (B, T, C) = predictions.shape
    predictions = predictions.view(B*T, C)
    y = y.view(-1)
    val_loss = nn.functional.cross_entropy(predictions, y)
    model.train()
    
    print(f'Step: {batch_index}. Train loss: {train_loss.item()}. Validation loss: {val_loss.item()}')
    

# Read input file.
filename = 'data/tinyshakespeare.txt'
data = read_dataset(filename)
(vocabulary, token_to_index_map, index_to_token_map, encoder, decoder) = create_vocabulary(data, False)
run_tokenizer_example(False)

# Tokenize dataset.
input_sequence = torch.tensor(encoder(data), dtype=torch.long)
#print(f'Tokenized input sequence is {input_sequence}.')
print(f'Shape: {input_sequence.shape}, Type: {input_sequence.dtype}.')

dataset_split_fraction = 0.9
num_train_samples = int(dataset_split_fraction * len(data))
train_set = input_sequence[:num_train_samples]
val_set = input_sequence[num_train_samples:]
print(f'Number of train samples is {len(train_set)}. Number of validation samples is {len(val_set)}.')



max_block_size = 24
batch_size = 32
num_batches = 1000
num_decoder_blocks = 10
vocabulary_size = len(vocabulary)
embedding_dimension = 64
num_heads = 8
head_dimension = 16


#model = BigramLanguageModel(len(vocabulary))
model = GPT(num_decoder_blocks, vocabulary_size, embedding_dimension, num_heads, head_dimension, max_block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch_index in range(num_batches):
    (x, y) = create_batch('train', max_block_size, batch_size)
    visualize_batch(x, y, True)
    predictions = model(x)

    (B, T, C) = predictions.shape
    predictions = predictions.view(B*T, C)
    y = y.view(-1)
    loss = nn.functional.cross_entropy(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if batch_index % 10 == 0:
        evaluate_loss(batch_index, model, batch_size, max_block_size)
        generated_tokens = model.generate(torch.zeros((1, 1), dtype = torch.long), 10).tolist()[0]
        generated_text = decoder(generated_tokens)
        print(f'Generated text is \n {generated_text}.')