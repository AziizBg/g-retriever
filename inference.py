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

    # Step 2: Load dataset and create test loader
    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()
    test_dataset = [dataset[i] for i in idx_split['test']]
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Step 3: Set up output directory and file path
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    print(f'path: {path}')

    # Step 4: Run inference using API
    progress_bar_test = tqdm(range(len(test_loader)))
    results = []
    for _, batch in enumerate(test_loader):
        with torch.no_grad():
            question = batch['question'][0]
            desc = batch['desc'][0] if 'desc' in batch else ""
            
            # Construct the prompt for WebQSP
            prompt = f"""Given the following knowledge graph information: {desc}
            Question: {question}
            Please provide a direct answer to the question based on the knowledge graph information above. The answer should be a single entity or a list of entities separated by '|'."""

            messages = [{
                "role": "system", 
                "content": "You are a knowledge graph question answering assistant. Your task is to answer questions based on the provided knowledge graph information. Provide direct answers without explanations."
            }, {
                "role": "user", 
                "content": prompt
            }]
            
            try:
                completion = client.chat.completions.create(
                    model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
                    messages=messages,
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=32,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stream=False
                )
                
                output = completion.choices[0].message.content.lower().strip()
                
                results.append({
                    "pred": output,
                    "label": batch['label'][0]
                })
                
            except Exception as e:
                print(f"Error during API call: {e}")
                results.append({
                    "pred": "error",
                    "label": batch['label'][0]
                })

        progress_bar_test.update(1)

    # Step 5: Save results and evaluate
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
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()