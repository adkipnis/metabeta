import torch
from tokenizer import FloatTokenizer
from transformer import Transformer
from data import Task, LinearModel

class Trainer:
    def __init__(self,
                 task: Task,
                 model: LinearModel,
                 transformer: Transformer,
                 n_samples: int = 1000,
                 n_epochs: int = 1000,
                 batch_size: int = 32,
                 lr: float = 0.001,
                 ):
        self.task = task
        self.model = model
        self.transformer = transformer
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()
            x = self.model.dataset(self.n_samples)
            beta = self.model.weights
            
            in_token = tokenizer.encodeTensor(x)
            in_tensor = torch.Tensor([tokenizer.sym_to_idx[sym] for sym in in_token])
            out_token = tokenizer.encode(beta)
            out_tensor = torch.Tensor([tokenizer.sym_to_idx[sym] for sym in out_token])

            # Forward pass
            output = self.transformer(in_tensor, out_tensor)
        
            # Calculate loss and backpropagate
            loss = criterion(output.view(-1, output.shape[-1]), tgt_expected.view(-1))
            loss.backward()
            
            optimizer.step()
        
            print(f"Epoch {epoch+1} Loss: {loss.item()}")

def main():
    task = Task(n_predictors=2)
    lin_model = LinearModel(task)
    tokenizer = FloatTokenizer()
    transformer = Transformer(tokenizer.n_symbols)
    trainer = Trainer(task, model, transformer)
    

if __name__ == '__main__':
    main()

