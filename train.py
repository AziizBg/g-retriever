import os
import wandb
import gc
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader
from openai import OpenAI
from src.config import parse_args_llama
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything

# Initialize NVIDIA client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

class NvidiaModelWrapper:
    """
    Wrapper for the NVIDIA Llama API model.

    Handles inference by sending batches of questions to the NVIDIA API and formatting
    the results for downstream evaluation.

    Attributes:
        args: Arguments containing model and inference configuration.
        temperature (float): Sampling temperature for the API.
        max_tokens (int): Maximum number of tokens to generate per response.
    """
    def __init__(self, args):
        """
        Initializes the NvidiaModelWrapper.

        Args:
            args: Namespace or dict containing model configuration, including max_new_tokens.
        """
        self.args = args
        self.temperature = 0.1
        self.max_tokens = args.max_new_tokens
        
    def inference(self, batch):
        """
        Performs inference on a batch of questions using the NVIDIA API.

        Args:
            batch (dict): Dictionary with 'question' and 'label' lists.

        Returns:
            list: List of dicts with 'pred' (prediction) and 'label' for each sample.
        """
        results = []
        for question, label in zip(batch['question'], batch['label']):
            messages = [{
                "role": "system",
                "content": "Analyze if these arguments support or counter each other. Respond with only 'support' or 'counter'."
            }, {
                "role": "user",
                "content": question
            }]
            
            try:
                completion = client.chat.completions.create(
                    model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                
                output = completion.choices[0].message.content.lower().strip()
                pred = 'support' if 'support' in output else 'counter'
                
                results.append({
                    "pred": pred,
                    "label": label
                })
                
            except Exception as e:
                print(f"API Error: {e}")
                results.append({
                    "pred": "error",
                    "label": label
                })
        return results

def main(args):
    """
    Main training and evaluation loop for the NVIDIA Llama API model.

    Handles:
        - Seeding and experiment tracking initialization.
        - Dataset loading and split.
        - DataLoader creation for validation and test sets.
        - Model wrapper initialization.
        - Validation loop with early stopping.
        - Final test evaluation and result saving.

    Args:
        args: Namespace or dict with experiment configuration.
    """
    # Initialize wandb
    seed_everything(seed=args.seed)
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_nvidia_llama_seed{args.seed}",
               config=args)
    print(args)

    # Load dataset
    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()
    
    # Filter out None samples to avoid AttributeError in collate_fn
    val_data = [dataset[i] for i in idx_split['val'] if dataset[i] is not None]
    test_data = [dataset[i] for i in idx_split['test'] if dataset[i] is not None]

    # Create data loaders
    val_loader = DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    # Initialize model wrapper
    model = NvidiaModelWrapper(args)

    # Validation phase (replaces training for API model)
    best_val_acc = 0.0
    best_epoch = 0
    
        
    for epoch in range(args.num_epochs):
        val_results = []
        
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            batch_results = model.inference(batch)
            val_results.extend(batch_results)
        
        # Check for empty val_results to avoid ZeroDivisionError
        if len(val_results) == 0:
            print("Warning: No validation results for this epoch. Skipping validation accuracy calculation.")
            continue

        val_acc = sum(1 for r in val_results if r['pred'] == r['label']) / len(val_results)
        wandb.log({'Val Acc': val_acc, 'epoch': epoch})
        print(f"Epoch {epoch}: Val Acc {val_acc:.4f}")

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            print(f"New best validation accuracy: {val_acc:.4f}")

        if epoch - best_epoch >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Final Evaluation
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/nvidia_llama_results_seed{args.seed}.csv'
    
    test_results = []
    for batch in tqdm(test_loader, desc="Testing"):
        batch_results = model.inference(batch)
        test_results.extend(batch_results)
    
    df = pd.DataFrame(test_results)
    df.to_csv(path, index=False)
    
    # Check for empty test_results to avoid ZeroDivisionError
    if len(test_results) == 0:
        print("Warning: No test results. Skipping test accuracy calculation.")
        test_acc = 0.0
    else:
        test_acc = sum(1 for r in test_results if r['pred'] == r['label']) / len(test_results)
        print(f'Final Test Accuracy: {test_acc:.4f}')
        wandb.log({'Test Acc': test_acc})
        
if __name__ == "__main__":
    """
    Entry point for running the script.

    - Parses command-line arguments.
    - Runs the main experiment loop.
    - Frees GPU and CPU memory after completion.
    """
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    gc.collect()