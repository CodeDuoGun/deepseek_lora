import evaluate
import torch

metric = evaluate.load("accuracy.py")
def compute_metrics(predictions, references, model, eval_dataloader,device="cuda"):
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            metric.compute()
