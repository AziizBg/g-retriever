# train2.py
import os
import wandb
import gc
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.model import load_model
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate

llama_model_path = {
    "mistral_7b": "mistralai/Mistral-7B-v0.1",
    "phi2": "microsoft/phi-2",
    # Add more as needed
}

def main(args):
    # Step 1: Set up wandb
    seed = args.seed
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_seed{seed}",
               config=args)

    seed_everything(seed=args.seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    train_dataset = [dataset[i] for i in idx_split['train'] if dataset[i] is not None]
    val_dataset = [dataset[i] for i in idx_split['val'] if dataset[i] is not None]
    test_dataset = [dataset[i] for i in idx_split['test'] if dataset[i] is not None]

    print('Train Dataset Sample:')
    try:
        print(train_dataset[0].keys())
    except AttributeError:
        print("Data is not a dictionary")
        print(train_dataset[0].__dict__.keys())

    # Reduce batch size and sequence length for memory efficiency
    args.batch_size = 1  # Further reduced from 2
    args.eval_batch_size = 1  # Further reduced from 2
    args.max_txt_len = 64  # Further reduced from 128
    args.max_new_tokens = 16
    args.grad_steps = 8  # Increased from 4 for more stable gradients

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # check args.llm_model_name
    if args.llm_model_name not in llama_model_path.keys():
        args.llm_model_name = "phi2"

    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]

    model = load_model[args.model_name](graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)

    # Step 4 Set Optimizer with reduced learning rate
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr * 0.000001, 'weight_decay': args.wd}],  # Further reduced learning rate
        betas=(0.9, 0.95)
    )
    
    # Add gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')
    best_epoch = -1

    # Add learning rate warmup
    warmup_steps = min(2000, len(train_loader))  # Increased warmup steps
    current_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0., 0.
        num_batches = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch

        for step, batch in enumerate(train_loader):
            # Clear GPU cache before each batch
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            # Validate and normalize input data
            for k, v in batch.items():
                if torch.is_tensor(v):
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        print(f"Bad values in {k}, skipping batch")
                        continue
                    # Normalize tensor values if they're not binary or categorical
                    if v.dtype in [torch.float32, torch.float16, torch.float64]:
                        v_norm = torch.norm(v, dim=-1, keepdim=True)
                        v_norm[v_norm == 0] = 1  # Prevent division by zero
                        v = v / v_norm

            # Use gradient scaling for mixed precision
            with torch.cuda.amp.autocast():
                try:
                    loss = model(batch)
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("NaN/Inf in loss, skipping batch")
                        continue
                        
                    # Scale loss to prevent overflow
                    loss = loss / args.grad_steps
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        print("GPU OOM, skipping batch")
                        continue
                    elif "indexing errors" in str(e):
                        print("Sequence length error, skipping batch")
                        continue
                    else:
                        raise e
                
            # Scale gradients and handle overflow
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % args.grad_steps == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients with reduced threshold
                grad_norm = clip_grad_norm_(optimizer.param_groups[0]['params'], 0.01)  # Further reduced from 0.05
                print({'Gradient Norm': grad_norm})
                
                # Skip update if gradients are NaN/Inf
                if grad_norm.isnan() or grad_norm.isinf():
                    print("NaN/Inf in gradient norm, skipping update")
                    optimizer.zero_grad()
                    continue
                    
                # Step optimizer with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Learning rate warmup with cosine schedule
                if current_step < warmup_steps:
                    lr_scale = 0.1 * (1 + torch.cos(torch.tensor(current_step / warmup_steps * torch.pi)))  # Reduced from 0.5
                    for pg in optimizer.param_groups:
                        pg['lr'] = args.lr * lr_scale
                else:
                    # Cosine decay after warmup
                    progress = (current_step - warmup_steps) / (num_training_steps - warmup_steps)
                    lr_scale = 0.1 * (1 + torch.cos(torch.tensor(progress * torch.pi)))  # Reduced from 0.5
                    for pg in optimizer.param_groups:
                        pg['lr'] = args.lr * lr_scale

                current_step += 1

            epoch_loss += loss.item() * args.grad_steps
            accum_loss += loss.item() * args.grad_steps
            num_batches += 1

            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({'Lr': lr, 'Accum Loss': accum_loss / args.grad_steps})
                accum_loss = 0.

            progress_bar.update(1)

        mean_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch}/{args.num_epochs} - Train Loss: {mean_train_loss:.4f}")
        wandb.log({'Train Loss (Epoch Mean)': mean_train_loss})

        # Evaluation
        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if any(torch.isnan(v).any() or torch.isinf(v).any() for v in batch.values() if torch.is_tensor(v)):
                    print(f"NaN/Inf detected in val batch {step}, skipping")
                    continue
                try:
                    loss = model(batch)
                    val_loss += loss.item()
                except RuntimeError as e:
                    if "out of memory" in str(e) or "indexing errors" in str(e):
                        print(f"Error in val batch {step}, skipping")
                        continue
                    raise e

        val_loss /= len(val_loader)
        print(f"Epoch {epoch}/{args.num_epochs} - Val Loss: {val_loss:.4f}")
        wandb.log({'Val Loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        if best_epoch == -1:
            print("Warning: No best epoch found.")
        else:
            print(f"Epoch {epoch} - Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Evaluation on test set
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    print(f'Saving inference to: {path}')

    model = _reload_best_model(model, args)
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                try:
                    output = model.inference(batch)
                    df = pd.DataFrame(output)
                    for _, row in df.iterrows():
                        f.write(json.dumps(dict(row)) + "\n")
                except RuntimeError as e:
                    if "out of memory" in str(e) or "indexing errors" in str(e):
                        print(f"Error in test batch {step}, skipping")
                        continue
                    raise e
            progress_bar_test.update(1)

    acc = eval_funcs[args.dataset](path)
    print(f'Test Accuracy: {acc}')
    wandb.log({'Test Acc': acc})


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()