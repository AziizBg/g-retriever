import os
import torch
import wandb
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import pandas as pd
from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_fn
from openai import OpenAI  # Import the OpenAI or NVIDIA client

# Initialize OpenAI or NVIDIA client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key="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"
)

def main(args):

    # Step 1: Set up wandb
    seed = args.seed
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_seed{seed}",
               config=args)

    seed_everything(seed=seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    test_dataset = [dataset[i] for i in idx_split['test']]
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Step 3: Replace model loading with OpenAI/NVIDIA API
    # No model loading needed here, because we'll call the API for inference

    # Step 4. Evaluating
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    print(f'path: {path}')

    progress_bar_test = tqdm(range(len(test_loader)))
    results = []
    for _, batch in enumerate(test_loader):
        with torch.no_grad():
            question = batch['question'][0]
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
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=10,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stream=False
                )
                
                output = completion.choices[0].message.content.lower().strip()
                pred = 'support' if 'support' in output else 'counter'
                
                results.append({
                    "pred": pred,
                    "label": batch['label'][0]
                })
                
            except Exception as e:
                print(f"Error during API call: {e}")
                results.append({
                    "pred": "error",
                    "label": batch['label'][0]
                })

        progress_bar_test.update(1)

    # Save results to CSV in the format expected by the evaluation function
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)

    # Evaluation
    try:
        acc = eval_funcs[args.dataset](path)
        print(f'Test Accuracy: {acc:.4f}')
        wandb.log({'Test Acc': acc})
    except Exception as e:
        print(f"Error during evaluation: {e}")
        wandb.log({'Test Acc': 0.0})

if __name__ == "__main__":
    args = parse_args_llama()
    main(args)

if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()