import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64 # how many independent sequences we will process in parallel?
block_size = 256 # what is the maximum context length for prediction
max_iters = 5000
eval_interval = 500 # evaluate the model every eval_interval iterations
lr = 3e-4 # learning rate, small since the model is large
eval_iter = 200
n_embed = 384 # meaning head_size = 384 // 6 -number of heads- = 64
n_layers = 6
n_heads = 6
dropout = 0.2 # 20% are dropped out

torch.manual_seed(1337)

# read the text file
with open("shakespear.txt","r",encoding='utf-8') as file:
        text = file.read()


# create a vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {char:i for i,char in enumerate(chars)}
itos = {i:char for i,char in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # takes a string, output list of integers
decode = lambda l: ''.join(itos[i] for i in l) # takes the list of integers and return the string


# tokenize the text
data = torch.tensor(encode(text),dtype=torch.long)
# train and validation split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,)) # the starting chunk index will be from 0 to len(data) - block_size (exclusive, so actually up to len(data) - block_size - 1, that is because we want block_size+1 to get the Xs and Ys
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train','val']:
        losses = torch.zeros(eval_iter)
        # we will loop over random eval_iter batches and average the losses to get a more robust estimate
        for k in range(eval_iter):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


## Model Components
class Head(nn.Module):
    """ One head of Self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        # the tril is used to mask out -discard- the upper triangular part of the weight matrix -the future tokens-
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size))) # trill is not a parameter, so it is called a buffer in pytorch naming conventions, so we have to assign it the module using the register_buffer method of the nn.Module class

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape 
        k = self.key(x) # B,T,head_size
        q = self.query(x) # B,T,head_size

        # compute the attention scores "Affinities"
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,head_size) @ (B,head_size,T) => (B,T,T)
        # discard the future tokens for each token
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        # apply softmax to get the attention weights
        wei = F.softmax(wei,dim=-1) # (B,T,T)

        # dropout some of the affinities randomly
        wei = self.dropout(wei)

        # get the values
        v = self.value(x) # B,T,head_size
        out = wei @ v # (B,T,T) @ (B,T,head_size) => (B,T,head_size)
        return out
    


class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        # pass the x to each head, result will be a list of tensors of shape (B,T,head_size), we concatenate them on the last dimension
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))
    

class FeedForward(nn.Module):
    """ a simple layer followed by a non-linearity """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 *n_embed),
            nn.ReLU(),
            # Project back into the residual path
            nn.Linear(4 * n_embed,n_embed),
            # dropout
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)
    



class block(nn.Module):
    """ Transformer Block: Communiaction followed by computation """

    def __init__(self, n_embed, num_heads):
        super().__init__()
        # we will make the head size so that the output of the multi-head attention has dimension n_embed
        head_size = n_embed // num_heads
        # the communication is done using multi-head attention
        self.self_attn = MultiHeadAttention(num_heads, head_size)
        # the computation is done using a feed forward layer
        self.ffwd = FeedForward(n_embed)
        # layer normalization (mine: we will use 2 different layer norms because we want to normalize the features of each token in the multi-head attention and the feed forward layer separately)
        self.ln1 = nn.LayerNorm(n_embed) # layer normalization before the multi-head attention 
        self.ln2 = nn.LayerNorm(n_embed) # layer normalization before the feed forward layer

    def forward(self,x):
        # communication (norm -> multi-head attention -> residual connection)
        x = x + self.self_attn(self.ln1(x))
        # computation (norm -> feed forward -> residual connection)
        x = x + self.ffwd(self.ln2(x))
        return x
    


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.positional_embedding_table = nn.Embedding(block_size,n_embed)
        self.blocks = nn.Sequential(*[block(n_embed,num_heads=n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed,vocab_size)
        

    def forward(self,idx,targets=None):
        """
        idx: the token indices, shape (batch_size,sequence_length)
        """
        B,T = idx.shape
        emb = self.token_embedding_table(idx) # embeddings of shape (batch_size,sequence_length,emb_size)
        pos_emb = self.positional_embedding_table(torch.arange(T,device=idx.device)) # (sequence_length,emb_size)
        x = emb + pos_emb # adding shapes (batch_size,sequence_length,emb_size) + (sequence_length,emb_size) -> broadcasting for the batch_size dimension
        
        # feed the input to the transformer blocks
        x = self.blocks(x) # (batch_size,sequence_length,emb_size)

        # layer normalization before the language model head
        x = self.ln_f(x)
        # feed the output of the blocks to the language model head
        logits = self.lm_head(x) # the logits of shape (batch_size,sequence_length,vocab_size)

        # if the targets are not provided (during generation)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.reshape(-1,vocab_size),targets.reshape(-1)) # we need to collapse the batch_size and the sequence length dimensions together (flatten out the timesteps as individual examples), that is what the loss expects
        
        return logits, loss

    def generate(self,idx,max_new_tokens):
        """
        idx: token indices of some batch (the same one used in training) (batch_size,sequence_length)
        we will basically take the indices and expand the sequence length using generation (sampling) up to max_new_tokens
        """
        for _ in range(max_new_tokens):
            # crop the sequence length to the block size
            idx_cropped = idx[:,-block_size:]
            # inference the idx
            logits, _ = self(idx_cropped) # batch_size,sequence_length,vocab_size
            # take the logits of the last token in the sequence
            logits = logits[:, -1, :] # becomes (batch_size, vocab_size)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=1) # still (batch_size, vocab_size), but each example in the batch now is betwee 0 and 1 and sums to 1
            # sample from the dsitribution
            idx_next = torch.multinomial(probs,num_samples=1) # batch_size,1, sampled next indices for each example in the batch
            # concatenate the sampled indices to the current indices (along the sequence length dimension)
            idx = torch.cat((idx,idx_next),dim=1) # batch_size, sequence_length + 1 = new sequence length
        return idx    


## Training
model = GPTLanguageModel().to(device)
# print the number of parameters in the model
print(f'{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# training loop
for i in range(max_iters):
    # get the batch
    x_batch,y_batch = get_batch('train')

    # Forward prop & loss 
    logits, loss = model(x_batch,y_batch)

    # backward prop
    # reset the gradients from the previous step before the backprop (we used to do it manually)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    ## update parameters
    optimizer.step()

    # validation phase each eval interval
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {i},  Train Loss: {losses['train']:.4f}, validation Loss: {losses['val']:.4f}")

# save the model
torch.save(model.state_dict(),'GPTLanguageModel.pth')
# Generation
context = torch.zeros(1,1,dtype=torch.long).to(device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))