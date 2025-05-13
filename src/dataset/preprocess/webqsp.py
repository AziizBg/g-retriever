import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'
data_dir = '/content/g-retriever/RoG-webqsp/data'

def load_parquet_dataset(data_dir):
    """
    Loads the WebQSP dataset splits from Parquet files and returns a Hugging Face DatasetDict.

    Args:
        data_dir (str): Path to the directory containing the Parquet files.

    Returns:
        DatasetDict: A dictionary with 'train', 'validation', and 'test' splits as Hugging Face Datasets.
            Each dataset contains columns as loaded from the Parquet files.
    """
    # Load each split separately using pandas
    train_df1 = pd.read_parquet(os.path.join(data_dir, 'train-00000-of-00002-d810a36ed97bc2cc.parquet'))
    train_df2 = pd.read_parquet(os.path.join(data_dir, 'train-00001-of-00002-e53244e71082a392.parquet'))
    train_df = pd.concat([train_df1, train_df2])
    
    val_df = pd.read_parquet(os.path.join(data_dir, 'validation-00000-of-00001-6ee6adc5b154643a.parquet'))
    
    test_df1 = pd.read_parquet(os.path.join(data_dir, 'test-00000-of-00002-9ee8d68f7d951e1f.parquet'))
    test_df2 = pd.read_parquet(os.path.join(data_dir, 'test-00001-of-00002-773a7b8213e159f5.parquet'))
    test_df = pd.concat([test_df1, test_df2])
    
    # Convert to Hugging Face datasets
    from datasets import Dataset
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })
    return dataset

def step_one():
    """
    Processes the WebQSP dataset to extract graph structures for each sample.
    For each sample, creates two CSV files:
        - nodes: mapping node IDs to node attributes (entity names)
        - edges: mapping source node, edge attribute (relation), and destination node

    The CSV files are saved in the directories specified by path_nodes and path_edges.
    """
    dataset = load_parquet_dataset(data_dir)
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        nodes = {}
        edges = []
        for tri in dataset[i]['graph']:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})
        nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        nodes.to_csv(f'{path_nodes}/{i}.csv', index=False)
        edges.to_csv(f'{path_edges}/{i}.csv', index=False)

def generate_split():
    """
    Generates and saves the train/validation/test split indices for the dataset.
    The indices are saved as text files in the split directory.
    Handles offsetting indices for validation and test splits to match concatenated dataset order.
    Removes known empty graph indices from validation.
    """
    dataset = load_parquet_dataset(data_dir)

    train_indices = np.arange(len(dataset['train']))
    val_indices = np.arange(len(dataset['validation'])) + len(dataset['train'])
    test_indices = np.arange(len(dataset['test'])) + len(dataset['train']) + len(dataset['validation'])

    # Remove empty graph indices
    val_indices = [i for i in val_indices if i != 2937]

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    os.makedirs(f'{path}/split', exist_ok=True)

    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))

import os
from concurrent.futures import ThreadPoolExecutor
import torch
from tqdm import tqdm

def step_two():
    print('Loading dataset...')
    dataset = load_parquet_dataset(data_dir)
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    
    # Initialize counters
    skipped = 0
    processed = 0
    total_graphs = len(dataset)
    
    # Configuration
    MAX_NODES = 1000  # Skip very large graphs
    SUBSAMPLE = False  # Set to True if you want to process only a subset
    SAMPLE_SIZE = 1000 if SUBSAMPLE else total_graphs

    questions = [i['question'] for i in dataset]

    # Initialize model
    model, tokenizer, device = load_model[model_name]()
    model = model.half().to(device)
    model.eval()
    text2embedding = load_text2embedding[model_name]

    print('Encoding questions...')
    q_embs = []
    batch_size = 64
    for i in tqdm(range(0, len(questions), batch_size)):
        batch = questions[i:i+batch_size]
        with torch.no_grad(), torch.cuda.amp.autocast():
            q_embs.append(text2embedding(model, tokenizer, device, batch))
    q_embs = torch.cat(q_embs)
    torch.save(q_embs, f'{path}/q_embs.pt')

    print(f'Encoding graphs (processing {SAMPLE_SIZE} of {total_graphs})...')
    os.makedirs(path_graphs, exist_ok=True)

    progress_bar = tqdm(total=SAMPLE_SIZE)
    for index in range(SAMPLE_SIZE):
        try:
            nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
            edges = pd.read_csv(f'{path_edges}/{index}.csv')
            nodes.node_attr = nodes.node_attr.fillna("")

            # Skip large graphs
            if len(nodes) > MAX_NODES:
                skipped += 1
                progress_bar.set_postfix({'skipped': skipped, 'last_size': len(nodes)})
                progress_bar.update(1)
                continue

            # Process graph
            with torch.no_grad(), torch.cuda.amp.autocast():
                x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
                edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            
            edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])
            pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
            torch.save(pyg_graph, f'{path_graphs}/{index}.pt')
            
            processed += 1
            progress_bar.set_postfix({'processed': processed, 'skipped': skipped})
            progress_bar.update(1)

        except Exception as e:
            print(f"\nError processing graph {index}: {str(e)}")
            continue

    progress_bar.close()
    print(f"\nCompleted! Processed: {processed}, Skipped: {skipped}, Error: {SAMPLE_SIZE - processed - skipped}")

if __name__ == '__main__':
    # step_one()
    # step_two()
    generate_split()