import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst
from src.dataset.preprocess.webqsp import load_parquet_dataset

model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'
data_dir = '/content/g-retriever/RoG-webqsp/data'


class WebQSPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = load_parquet_dataset(data_dir)
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        # skip if files do not exist
        if not os.path.exists(f'{cached_graph}/{index}.pt') or not os.path.exists(f'{cached_desc}/{index}.txt'):
            print(f'Graph file does not exist at index {index}')
            return None
        graph = torch.load(f'{cached_graph}/{index}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


def preprocess():
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    dataset = load_parquet_dataset(data_dir)
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    skipped = 0
    q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(dataset))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue

        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        # skip if files do not exist
        if not os.path.exists(f'{path_graphs}/{index}.pt') or not os.path.exists(f'{path_nodes}/{index}.csv') or not os.path.exists(f'{path_edges}/{index}.csv'):
            # print(f'Graph file does not exist at index {index}')
            skipped += 1
            continue
        graph = torch.load(f'{path_graphs}/{index}.pt')
        q_emb = q_embs[index]
        subg, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)
        torch.save(subg, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':

    preprocess()

    dataset = WebQSPDataset()

    # look for 5 items that are not None and print them out
    j = 0
    for i in range(len(dataset)):
        if j == 5:
            break
        data = dataset[i]
        if data is not None:
            print(f'Index: {i}')
            for k, v in data.items():
                print(f'{k}: {v}')
            split_ids = dataset.get_idx_split()
            for k, v in split_ids.items():
                print(f'# {k}: {len(v)}')
            j += 1
            break