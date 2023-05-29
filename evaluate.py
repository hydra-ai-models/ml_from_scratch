# Functions for evaluating language models.

import torch

class Evaluator:
    @torch.no_grad()
    def evaluate_train_and_validation_loss(self, train_dataloader, val_dataloader, model, num_batches_to_evaluate, device):
        model.eval()
        train_loss = self.evaluate_loss(train_dataloader, model, num_batches_to_evaluate, device)
        val_loss = self.evaluate_loss(val_dataloader, model, num_batches_to_evaluate, device)
        model.train()
        return (train_loss, val_loss)

    @torch.no_grad()
    def evaluate_loss(self, dataloader, model, num_batches_to_evaluate, device):
        loss = torch.zeros(num_batches_to_evaluate)
        for (batch_index, batch) in enumerate(dataloader):
            if batch_index >= num_batches_to_evaluate:
                break
            current_features = batch['features'].to(device)
            current_labels = batch['labels'].to(device)

            _, current_loss = model(current_features, current_labels)
            loss[batch_index] = current_loss
        return loss.mean()

    @torch.no_grad()
    def generate_text(self, model, num_tokens_to_generate, tokenizer, device) -> str:
        initial_token = torch.zeros((1, 1), device = device, dtype = torch.long)
        generated_tokens = model.generate(initial_token, num_tokens_to_generate)
        return tokenizer.decode(generated_tokens.tolist()[0])
