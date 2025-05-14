import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst
from src.dataset.preprocess.webqsp import load_parquet_dataset

# === BASE PATHS DANS GOOGLE DRIVE ===
drive_path = '/content/drive/MyDrive/Projets TPs GL4/PFA GL4/webqsp'  # or your desired persistent directory
os.makedirs(drive_path, exist_ok=True)

model_name = 'sbert'
path = os.path.join(drive_path, 'dataset/webqsp')
path_nodes = os.path.join(path, 'nodes')
path_edges = os.path.join(path, 'edges')
path_graphs = os.path.join(path, 'graphs')

cached_graph = os.path.join(path, 'cached_graphs')
cached_desc = os.path.join(path, 'cached_desc')
split_path = os.path.join(path, 'split')

data_dir = '/content/g-retriever/RoG-webqsp/data'  # <-- peut rester local pour traitement temporaire


class WebQSPDataset(Dataset):
    def __init__(self, sample_size=50):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = load_parquet_dataset(data_dir)
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']]).select(range(sample_size))
        self.q_embs = torch.load(os.path.join(path, 'q_embs.pt'))
        self.sample_size = sample_size

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph_file = os.path.join(cached_graph, f'{index}.pt')
        desc_file = os.path.join(cached_desc, f'{index}.txt')
        if not os.path.exists(graph_file) or not os.path.exists(desc_file):
            print(f'Graph file does not exist at index {index}')
            return None
        graph = torch.load(graph_file)
        desc = open(desc_file, 'r').read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):
        with open(os.path.join(split_path, 'train_indices.txt'), 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(os.path.join(split_path, 'val_indices.txt'), 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(os.path.join(split_path, 'test_indices.txt'), 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


def preprocess():
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    dataset = load_parquet_dataset(data_dir)
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    skipped = 0
    SUBSAMPLE = True
    SAMPLE_SIZE = 50 if SUBSAMPLE else len(dataset)

    q_embs = torch.load(os.path.join(path, 'q_embs.pt'))
    for index in range(SAMPLE_SIZE):
        graph_file = os.path.join(cached_graph, f'{index}.pt')
        desc_file = os.path.join(cached_desc, f'{index}.txt')

        if os.path.exists(graph_file):
            continue

        try:
            nodes = pd.read_csv(os.path.join(path_nodes, f'{index}.csv'))
            edges = pd.read_csv(os.path.join(path_edges, f'{index}.csv'))
            graph = torch.load(os.path.join(path_graphs, f'{index}.pt'))
        except FileNotFoundError:
            skipped += 1
            continue

        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue

        q_emb = q_embs[index]
        subg, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)
        torch.save(subg, graph_file)
        with open(desc_file, 'w') as f:
            f.write(desc)


if __name__ == '__main__':
    preprocess()

    dataset = WebQSPDataset()

    data = dataset[0]
    print(f'Index: {0}')
    for k, v in data.items():
        print(f'{k}: {v}')
    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
