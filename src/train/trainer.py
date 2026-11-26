import torch

class Trainer:

    def __init__(self, model, optimizer, criterion, dataloader, cfg):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        self.model.train()

        for epoch in range(self.cfg["train"]["epochs"]):
            total_loss = 0

            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            
                self.optimizer.zero_grad()
            
                outputs = self.model(**batch)
                loss = self.criterion(outputs, batch["labels"])
            
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")