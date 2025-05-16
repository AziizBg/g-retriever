# graph_llm.py
import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
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
            use_cache=False,  # Disable cache to save memory
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        self.word_embedding = self.model.model.get_input_embeddings()
        llm_embed_dim = self.word_embedding.embedding_dim  # Get LLM embedding size dynamically

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)


        # Replace your projector with this more stable version
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.LayerNorm(2048),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.GELU(),
            nn.Linear(2048, llm_embed_dim),
            nn.LayerNorm(llm_embed_dim)  # Final normalization
        ).to(self.model.device)

        # Initialize with smaller weights
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # 'gelu' not supported, use 'relu'
                nn.init.zeros_(layer.bias)
        
        # Add gradient clipping hooks
        for p in self.projector.parameters():
            p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=1e4, neginf=-1e4))


    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):  # <-- change default to float16
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def encode_graphs(self, samples):
        graphs = samples['graph']
        # Check for missing or empty graphs
        if graphs is None or not hasattr(graphs, 'x') or graphs.x is None or graphs.x.numel() == 0:
            print("Skipping batch due to missing or empty graph data.")
            return torch.tensor(float('nan'), device=self.device)

        graphs = graphs.to(self.model.device)
        n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)
        # Clamp or replace NaNs/Infs
        if torch.isnan(n_embeds).any() or torch.isinf(n_embeds).any():
            print("NaN/Inf in n_embeds from graph_encoder")
            print("graphs.x:", graphs.x)
            print("graphs.edge_index:", graphs.edge_index)
            print("graphs.edge_attr:", graphs.edge_attr)
            n_embeds = torch.nan_to_num(n_embeds, nan=0.0, posinf=1e4, neginf=-1e4)
        # mean pooling
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
        return g_embeds

    def forward(self, samples):
        # Add input validation
        for k, v in samples.items():
            if torch.is_tensor(v):
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(f"NaN/Inf detected in input {k}")
                    return torch.tensor(float('nan'), device=self.device)
        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        input_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.word_embedding.weight.device)
        max_model_len = getattr(self.model.config, "max_position_embeddings", 2048)
        input_ids = input_ids[-max_model_len:]  # Truncate from the left if too long
        bos_embeds = self.word_embedding(input_ids)        
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id).to(self.word_embedding.weight.device)
        ).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)
        

        # Add embedding validation
        if torch.isnan(graph_embeds).any() or torch.isinf(graph_embeds).any():
            print("NaN/Inf in graph_embeds")
            return torch.tensor(float('nan'), device=self.device)
        
        # print("graph_embeds.shape after projection:", graph_embeds.shape)
        # print("graph_embeds[0].shape after projection:", graph_embeds[0].shape)
        # print("bos_embeds.shape:", bos_embeds.shape)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
                print("NaN/Inf in inputs_embeds")
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id).to(self.word_embedding.weight.device)
        ).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # STRICTLY TRUNCATE INPUTS
            max_allowed_desc = self.max_txt_len - len(questions.input_ids[i]) - len(eos_user_tokens.input_ids) - len(label_input_ids)
            desc_ids = descriptions.input_ids[i][:max_allowed_desc]
            
            input_ids = desc_ids + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
