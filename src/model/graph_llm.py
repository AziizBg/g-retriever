# graph_llm.py
import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            "max_memory": {0: '80GiB'},
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=False,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        self.word_embedding = self.model.model.get_input_embeddings()
        llm_embed_dim = self.word_embedding.embedding_dim

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.LayerNorm(2048),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(2048, llm_embed_dim),
            nn.LayerNorm(llm_embed_dim)
        ).to(self.model.device)

        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)

        for p in self.projector.parameters():
            p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=1e4, neginf=-1e4))

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        return autocast(dtype=dtype) if self.device != torch.device("cpu") else contextlib.nullcontext()

    def encode_graphs(self, samples):
        graphs = samples['graph']
        if graphs is None or not hasattr(graphs, 'x') or graphs.x is None or graphs.x.numel() == 0:
            print("Skipping batch due to missing or empty graph data.")
            return None

        graphs = graphs.to(self.model.device)
        n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)

        if torch.isnan(n_embeds).any() or torch.isinf(n_embeds).any():
            print("NaN/Inf in n_embeds from graph_encoder")
            print("graphs.x:", graphs.x)
            print("graphs.edge_index:", graphs.edge_index)
            print("graphs.edge_attr:", graphs.edge_attr)
            return None

        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
        return g_embeds

    def forward(self, samples):
        for k, v in samples.items():
            if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
                print(f"NaN/Inf detected in input {k}")
                return torch.tensor(float('nan'), device=self.device)

        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        input_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device)
        bos_embeds = self.word_embedding(input_ids)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        graph_embeds = self.encode_graphs(samples)
        if graph_embeds is None:
            return torch.tensor(float('nan'), device=self.device)

        graph_embeds = self.projector(graph_embeds)
        if torch.isnan(graph_embeds).any() or torch.isinf(graph_embeds).any():
            print("NaN/Inf in graph_embeds after projector")
            return torch.tensor(float('nan'), device=self.device)

        batch_size = len(samples['id'])
        batch_inputs_embeds, batch_attention_mask, batch_label_input_ids = [], [], []

        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

            if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
                print("NaN/Inf in inputs_embeds")
                continue

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_len = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_len, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_len + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss
