"""
pyexample.

An example python library.
"""

__version__ = "0.0.1"
__author__ = 'Shourya Goel'
__credits__ = 'VLG'

from dataclasses import dataclass
from typing import Optional, Iterator, Dict, List
import json
from pathlib import Path
import gc
import argparse
from typing_extensions import Literal
import warnings
import inspect
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import os
from rouge_score import rouge_scorer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging
)

from datasets import (
    load_dataset,
    IterableDataset
)

logging.set_verbosity_error()

@dataclass
class ModelConfig:
    model_name: str
    model_family: str
    model_path: Optional[Path]

@dataclass
class QuantConfig:
    level: Optional[int]
    load_in_4bit: Optional[bool]
    load_in_8bit: Optional[bool]
    bnb_4bit_use_double_quant: Optional[bool]
    bnb_4bit_quant_storage: Optional[torch.dtype]
    bnb_4bit_compute_dtype: Optional[torch.dtype]
    bnb_4bit_quant_type: Optional[str] = None

@dataclass
class LayerSwapConfig:
    skip_layers: Optional[List[int]]

def create_dtype_map() -> Dict[str, torch.device]:
    mapping = {
        ("float16", "fp16") : torch.float16,
        ("bfloat16",)       : torch.bfloat16,
        ("float32", "fp32") : torch.float32,
    }
    dtype_map = {}
    for keys, value in mapping.items():
        for key in keys:
            dtype_map[key] = value
    return dtype_map


def load_config_from_json(
    json_file: Path,
    config_type: Literal["model", "quant", "layer_swap"]
) -> Iterator[QuantConfig]:
    with open(json_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if config_type == "model":
        for config in configs:
            yield ModelConfig(**config)
    elif config_type == "quant":
        for config in configs:
            yield QuantConfig(**config)
    elif config_type == "layer_swap":
        for config in configs:
            yield LayerSwapConfig(**config)

def generate_decoder_map():
    decoder_map: Dict[str, str] = {
        "llama": "model.layers",
        "gpt_neox": "gpt_neox.layers",
        "mistral": "model.layers",
        "mixtral": "model.layers",
    }
    return decoder_map

DecoderMap = generate_decoder_map()

def set_seed(seed: int = 42):
    random.seed(seed)                          # Python random module
    np.random.seed(seed)                       # NumPy
    torch.manual_seed(seed)                    # PyTorch CPU
    torch.cuda.manual_seed(seed)               # PyTorch GPU
    torch.cuda.manual_seed_all(seed)           # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # Avoid non-deterministic optimizations

class MemorizationAnalyser:
    def __init__(
        self,
        model_config: ModelConfig,
        quant_config: QuantConfig,
        quant_config_swap: Optional[QuantConfig],
        layer_swap_config: Optional[LayerSwapConfig],
        swap_every: Optional[List[str]],
        dataset_name: str = "legacy-datasets/wikipedia",
        batch_size: int = 128,
        device_map: Literal["cpu", "auto", "balanced"] = "balanced",
        dtype_map: Dict = create_dtype_map(),
    ):
        if layer_swap_config is not None and swap_every is not None:
            raise ValueError(f"Please specify only one of layer_swap_config or swap_every")
        self.model_name = model_config.model_name
        self.dataset_name = dataset_name
        self.dataset = None
        self.batch_size = batch_size
        self.device_map = device_map
        
        self.dtype_map = dtype_map
        self.dtype = self.dtype_map[quant_config.bnb_4bit_compute_dtype]
        
        self.load_in = "fp32"
        if quant_config.load_in_4bit:
            self.load_in = "4bit/" +  quant_config.bnb_4bit_quant_type
            if quant_config.bnb_4bit_use_double_quant:
                self.load_in += "/double"
            else:
                self.load_in += "/single" 
                
        elif quant_config.load_in_8bit:
            self.load_in = "8bit"
        
        if quant_config.bnb_4bit_quant_type:
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit = quant_config.load_in_4bit,
                bnb_4bit_quant_type = quant_config.bnb_4bit_quant_type,
                load_in_8bit = quant_config.load_in_8bit,
                bnb_4bit_use_double_quant = quant_config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype = self.dtype
            )
        else:
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit = quant_config.load_in_4bit,
                load_in_8bit = quant_config.load_in_8bit,
                bnb_4bit_use_double_quant = quant_config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype = self.dtype
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            use_fast=True,
            clean_up_tokenization_spaces=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config = self.quant_config,
            device_map=self.device_map
        )
        self.model.eval()
        self.context_length = 2000
        self.log_path = (
            f"./logs/model={self.model_name}/compute_dtype={self.dtype}/"
            f"load_in={self.load_in}"
        )
        os.makedirs(self.log_path, exist_ok=True)

        if quant_config_swap and (layer_swap_config or swap_every):
            self.dtype_swap = self.dtype_map[quant_config_swap.bnb_4bit_compute_dtype]
            
            if quant_config.bnb_4bit_quant_type:
                self.quant_config_swap = BitsAndBytesConfig(
                    load_in_4bit = quant_config.load_in_4bit,
                    bnb_4bit_quant_type = quant_config.bnb_4bit_quant_type,
                    load_in_8bit = quant_config.load_in_8bit,
                    bnb_4bit_use_double_quant = quant_config.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype = self.dtype
                )
            else:
                self.quant_config_swap = BitsAndBytesConfig(
                    load_in_4bit = quant_config.load_in_4bit,
                    load_in_8bit = quant_config.load_in_8bit,
                    bnb_4bit_use_double_quant = quant_config.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype = self.dtype
                )
            # self.quant_config_swap = BitsAndBytesConfig(
            #     load_in_4bit = quant_config_swap.load_in_4bit,
            #     bnb_4bit_quant_type = quant_config_swap.bnb_4bit_quant_type,
            #     load_in_8bit = quant_config_swap.load_in_8bit,
            #     bnb_4bit_use_double_quant = quant_config_swap.bnb_4bit_use_double_quant,
            #     bnb_4bit_compute_dtype = self.dtype_swap
            # )
            
            if quant_config_swap.load_in_4bit:
                self.load_in_swap = "4bit/" +  quant_config_swap.bnb_4bit_quant_type
                if quant_config_swap.bnb_4bit_use_double_quant:
                    self.load_in_swap += "/double"
                else: self.load_in_swap += "/single"
                
            elif quant_config_swap.load_in_8bit:
                self.load_in_swap = "8bit"
                
            self.model_swap = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config = self.quant_config_swap,
                device_map=self.device_map
            )
            self.model_swap.eval()
            self.decoder_map = generate_decoder_map()
            
            try:
                decoders_1 = eval(f"self.model.{DecoderMap[model_config.model_family]}")
                decoders_2 = eval(f"self.model_swap.{DecoderMap[model_config.model_family]}")
            except AttributeError as e:
                    raise ValueError(f"Unsupported model family '{model_config.model_family}' \
                        or invalid layer attribute: {e}")

            if layer_swap_config:
                self.layer_swap_config = layer_swap_config
                self.log_path = (
                    f"./logs/model={self.model_name}/compute_dtype={self.dtype}/"
                    f"load_in={self.load_in}/"
                    f"quantize_specific/swap_dtype={self.dtype_swap}/"
                    f"load_in_swap={self.load_in_swap}/"
                    f"layer_swap_config={layer_swap_config}"
                )
                os.makedirs(self.log_path, exist_ok=True)
                for layer in self.layer_swap_config.skip_layers:
                    decoders_1[layer] = decoders_2[layer]
                         
            elif swap_every:
                print(f"swap_every: {swap_every}")
                self.swap_every = swap_every
                self.log_path = (
                    f"./logs/model={self.model_name}/compute_dtype={self.dtype}/"
                    f"load_in={self.load_in}/"
                    f"quantize_specific/swap_dtype={self.dtype_swap}/"
                    f"load_in_swap={self.load_in_swap}/"
                    f"swap_every={'_'.join(s.replace(' ', '_').replace('/', '%') for s in swap_every)}"
                )
                print(f"log_path: {self.log_path}")
                # print(f"log_path={self.log_path}")
                os.makedirs(self.log_path, exist_ok=True)
                for swap in self.swap_every:
                    try:
                        swap = swap.strip()
                        for swap in swap.split():
                            num, denom = map(int, swap.split("/"))
                            for layer in range(len(decoders_1)):
                                if (layer + 1) % denom == num % denom:
                                    decoders_1[layer] = decoders_2[layer]
                    except ValueError as e:
                        raise ValueError("swap_every must be in the format 'x/y' \
                                        where x and y are positive integers: {e}")

                        
        if (quant_config_swap is None and (layer_swap_config is not None or swap_every is not None)) or \
            (quant_config_swap is not None and layer_swap_config is None and swap_every is None):
            raise ValueError(f"Please provide both quant_config_swap and either layer_swap_config or swap_every,\
                            but not both.")
        
    def get_completion(
        self,
        # max_new_tokens: int = 50,
        context_lengths: List[float]= [0.025, 0.05, 0.1, 0.25],
        target_lengths: List[float] = [0.025, 0.05, 0.1, 0.25],
        num_samples: int = 1000,
        baseline_memorized: Optional[Path] = None,
    ):  
        context_lengths = [int(context_length * self.context_length) for context_length in context_lengths]
        target_lengths = [int(target_length * self.context_length) for target_length in target_lengths]
        measure_rouge = rouge_scorer.RougeScorer([
            'rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        results: List = []
        results_summary: Dict = {}
        if baseline_memorized is not None:
            with open(baseline_memorized, "r") as f:
                json_data = json.load(f)
                            
        for context_length in tqdm(context_lengths, desc="Context Lengths"):
            for target_length in tqdm(target_lengths, desc=f"Processing Context Length: {context_length}", leave=False):
                print(f"Model: {self.model_name}, Dataset: {self.dataset_name}, \
                        Context Length: {context_length}, Target Length: {target_length}")
                self.memorized: int = 0
                self.memorized_list: List[int] = []
                self.outlier_list: List[int] = []
                self.prompts: int = 0
                for i, prompts in enumerate(tqdm(self.dataset, desc=f"Processing Context Length: {context_length}, Target Length: {target_length}", leave=False)): 
                    
                    inputs = self.tokenizer(
                        prompts["text"],
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length= max(context_lengths)+target_length,
                    ).to(self.model.device)
                    
                    # print(f"inputs: {inputs}")
                    
                    prompt_tokens = inputs["input_ids"][:, :context_length]
                    attention_mask = inputs["attention_mask"][:, :context_length]
                    target_tokens = inputs["input_ids"][:, context_length:context_length + target_length]
                    
                    # print(f"target tokens: {target_tokens}")
                    # print(f"prompt tokens: {prompt_tokens}")
                    # print(f"prompt tokens attention mask: {attention_mask}")
                    
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            prompt_tokens,
                            attention_mask=attention_mask,
                            max_new_tokens=target_length,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                        
                        # print(f"output_ids: {output_ids}")
                        # print(f"decoded output: {self.tokenizer.batch_decode(output_ids)}")
                    
                    # print((target_tokens == output_ids[:, context_length:context_length + target_length]).all(dim=1).sum().item())
                    # print(target_tokens.size())
                    # self.memorized += (target_tokens == output_ids[:, context_length:context_length + target_length]).all(dim=1).sum().item()
                    # self.prompts += len(output_ids)
                    # print(target_tokens.shape)
                    # print(output_ids.shape)
                    # exit()
                    memorized_mask = (target_tokens == output_ids[:, context_length:context_length + target_length]).all(dim=1)  
                    self.memorized += memorized_mask.sum().item()
                    key = f"Model={self.model_name}_Context={context_length}_Target={target_length}"
                    print(f"key: {key}")
                    if baseline_memorized is not None:
                        for data in json_data:
                            if key in data:
                                baseline_memorized_ = data[key]["Index List"]     
                    else:
                        baseline_memorized_ = []
                    
                    # print(f"memorized_mask: {memorized_mask}")
                    # results: List = []
                    for idx, is_memorized in enumerate(memorized_mask):
                        # print(f"idx: {idx}")
                        # print(f"self.prompts + idx: {self.prompts + idx}")
                        decoded_target = self.tokenizer.decode(
                            target_tokens[idx],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True    
                        )
                        decoded_output = self.tokenizer.decode(
                            output_ids[idx],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True  
                        )
                        if is_memorized and idx in baseline_memorized_:
                            self.memorized_list.append(self.prompts + idx)  
                            # print(f"  \
                            #     Index: {self.prompts + idx} \
                            #     Target: {target_tokens} \
                            #     Decoded Target: {decoded_target} \
                            #     Decoded Output: {decoded_output} \
                            #     Context Length: {context_length} \
                            #     Target Length: {target_length} \
                            #     Memorized: {is_memorized} \
                            #     Baseline Memorized: {bool(idx in baseline_memorized_)} \
                            #         ")
                            results.append({
                                "Index": self.prompts + idx,  
                                "Target": target_tokens[idx].tolist(), # Tensor -> List
                                "Decoded Target": decoded_target,  
                                "Decoded Output": decoded_output,  
                                "Context Length": context_length,
                                "Target Length": target_length,
                                "Memorized": is_memorized,
                                "Baseline Memorized": bool(idx in baseline_memorized_)
                            })
                        elif is_memorized and idx not in baseline_memorized_:
                            # print(
                            #     f"idx: {idx} memorized but not memorized by baseline model \
                            #     Maybe a baseline memorized list was not provided. \
                            #     Let's check.. baseline memorized list present: {bool(baseline_memorized)}"
                            # )
                            # print(f"  \
                            #     Index: {self.prompts + idx} \
                            #     Target: {target_tokens} \
                            #     Decoded Target: {decoded_target} \
                            #     Decoded Output: {decoded_output} \
                            #     Context Length: {context_length} \
                            #     Target Length: {target_length} \
                            #     Memorized: {is_memorized} \
                            #     Baseline Memorized: {bool(idx in baseline_memorized_)} \
                            #         ")
                            results.append({
                                "Index": self.prompts + idx,  
                                "Target": target_tokens[idx].tolist(), # Tensor -> List
                                "Decoded Target": decoded_target,  
                                "Decoded Output": decoded_output,  
                                "Context Length": context_length,
                                "Target Length": target_length,
                                "Memorized": is_memorized,
                                "Baseline Memorized": bool(idx in baseline_memorized_)
                            })
                            self.memorized_list.append(self.prompts + idx)
                            self.outlier_list.append(self.prompts + idx)
                        
                        elif not is_memorized and idx in baseline_memorized_:
                            # gibberish logic
                            # print(f"  \
                            #     Index: {self.prompts + idx} \
                            #     Target: {target_tokens} \
                            #     Decoded Target: {decoded_target} \
                            #     Decoded Output: {decoded_output} \
                            #     Context Length: {context_length} \
                            #     Target Length: {target_length} \
                            #     Memorized: {is_memorized} \
                            #     Baseline Memorized: {bool(idx in baseline_memorized_)} \
                            #     Rogue Scores: {measure_rouge.score(decoded_target, decoded_output)} \
                            #         ")
                            results.append({
                                "Index": self.prompts + idx,  
                                "Target": target_tokens[idx].tolist(), # Tensor -> List
                                "Decoded Target": decoded_target, 
                                "Decoded Output": decoded_output,  
                                "Context Length": context_length,
                                "Target Length": target_length,
                                "Memorized": is_memorized,
                                "Baseline Memorized": bool(idx in baseline_memorized_),
                                "Rogue Scores": measure_rouge.score(decoded_target, decoded_output)
                            })
                            
                            
                    
                    self.prompts += len(output_ids) # len(memorized_mask) would work too!
                    # print(f"self.prompts: {self.prompts}")
                    
                    # print(f"prompt tokens device: {prompt_tokens.device}")
                    # print(f"target tokens device: {target_tokens.device}")
                    # print(f"attention_mask device: {prompt_tokens.device}")
                    # print(f"inputs device: {inputs.input_ids.device}")
                    
                    if self.prompts % 1_000 == 0:
                        df = pd.DataFrame(results)
                        df.to_csv(f"{self.log_path}/results.csv", index=False)
                        
                    elif self.prompts >= num_samples: 
                        print(f"Memorized: {self.memorized / self.prompts}")
                        results.append({
                            "Index": "Final Summary",  
                            "Context Length": context_length,
                            "Target Length": target_length,
                            "Memorized": f"Total Memorized: {self.memorized / self.prompts}, Index List: {self.memorized_list}",
                        })
                        
                        df = pd.DataFrame(results)
                        df.to_csv(f"{self.log_path}/results.csv", index=False)
                        
                        results_summary[f"Model={self.model_name}_Context={context_length}_Target={target_length}"] = {
                            "Context Length": context_length,
                            "Target Length": target_length,
                            "Memorized Ratio": self.memorized / self.prompts,
                            "Index List": self.memorized_list,
                            "Outlier List": self.outlier_list,
                            "Total Samples": self.prompts
                        }
                        
                        with open(f"{self.log_path}/summary.json", "w") as json_file:
                            json.dump(results_summary, json_file, indent=4)
                        
                        break
                    
                    del inputs, prompt_tokens, target_tokens, attention_mask, output_ids 
                    gc.collect()
                
                    
    def sample_from_pile(self, min_length=100, batch_size=128, seed=42):
        
        self.dataset : IterableDataset = load_dataset(
            self.dataset_name,
            "20220301.en",
            split="train",
            streaming=True,
            trust_remote_code=True
        ).shuffle(
            seed  
        ).filter(
            lambda prompt: len(prompt["text"].split()) >= min_length
        ).batch(
            batch_size
        )

def run_analysis(
    model_config: ModelConfig,
    quant_config: QuantConfig,
    dataset_name: str,
    batch_size: int,
    device_map: str,
    max_new_tokens: int,
    context_lengths: List[int],
    target_lengths: List[int],
    num_samples: int,
    quant_config_swap: Optional[QuantConfig] = None,
    layer_swap_config: Optional[LayerSwapConfig] = None,
    swap_every: Optional[int] = None,
    baseline_memorized: Optional[Path] = None
) -> None:
    
    analyzer = MemorizationAnalyser(
        model_config=model_config,
        quant_config=quant_config,
        quant_config_swap=quant_config_swap,
        layer_swap_config=layer_swap_config,
        swap_every=swap_every,
        dataset_name=dataset_name,
        batch_size=batch_size,
        device_map=device_map,
    )
    
    print(f"Loaded model: {analyzer.model}")
    
    # Prepare the dataset
    analyzer.sample_from_pile(batch_size=batch_size)
    
    # Run the analysis
    analyzer.get_completion(
        # max_new_tokens=max_new_tokens,
        context_lengths=context_lengths,
        target_lengths=target_lengths,
        num_samples=num_samples,
        baseline_memorized=baseline_memorized
    )
    
    # Cleanup
    del analyzer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Analyze degree of memorization for a specified model and dataset, \
            varying quantization parameters"
    )
    parser.add_argument(
        "--model-config", 
        type=str,
        required = True,
        help = "Path to the model JSON configuration file"
    )
    parser.add_argument(
        "--quant-config", 
        type = str,
        required = True,
        help = "Path to the quantization JSON configuration file"
    )
    parser.add_argument(
        "--quant-config-swap", 
        type = str,
        help = "Path to the swap quantization JSON configuration file"
    )
    parser.add_argument(
        "--layer-swap-config", 
        type = str,
        help = "Path to the layer swap JSON configuration file"
    )
    parser.add_argument(
        "--baseline-memorized",
        type = str,
        help = "Baseline memorized dataset index list"
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default= "legacy-datasets/wikipedia",
        help = "Benchmark dataset name"
    )
    parser.add_argument(
        "--num-samples",
        type = int,
        default= 100,
        help = "Number of samples analyzed from dataset"
    )
    parser.add_argument(
        "--batch-size",
        type = int,
        default= 128,
        help = "Batch size"
    )
    parser.add_argument(
        "--swap-every",
        type = str,
        nargs="+",
        default=["3/4", "4/4"],
        help = "Specify which fraction of decoder layers to quantize e.g ['3/4', '4/4'] \
                will quantize the third and fourth quarter of decoders"
    )
    parser.add_argument(
        "--device-map",
        type = str,
        default="auto",
        help = "BitsAndBytes Device Map configuration"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate during analysis"
    )
    parser.add_argument(
        "--context-lengths",
        type=float,
        nargs="+",
        default=[0.025, 0.05, 0.1, 0.25],
        help="List of context lengths to analyze"
    )
    parser.add_argument(
        "--target-lengths",
        type=float,
        nargs="+",
        default=[0.025, 0.05, 0.1, 0.25],
        help="List of target lengths to analyze"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="common fixed seed for all libraries"
    )

    args = parser.parse_args()
    set_seed(args.seed)
    model_configs = load_config_from_json(Path(args.model_config), config_type="model")
    for model_config in model_configs:
        print(f"Model: {model_config}")
        quant_configs = load_config_from_json(Path(args.quant_config), config_type="quant")
        for quant_config in quant_configs:
            print(f"Quantization Config: {quant_config}")
            base_level = quant_config.level
            
            print(f"args.quant_config_swap: {args.quant_config_swap}") 
            if args.quant_config_swap is not None:
                quant_config_swaps = load_config_from_json(
                    Path(args.quant_config_swap),
                    config_type="quant"
                )
                
                for quant_config_swap in quant_config_swaps:
                    swap_level = quant_config_swap.level # do not cover previous cases
                    if swap_level <= base_level:
                        continue
                    
                    if args.layer_swap_config:
                        layer_swap_configs = load_config_from_json(
                            Path(args.layer_swap_config), 
                            config_type="layer_swap"
                        )
                
                        for layer_swap_config in layer_swap_configs:
                                run_analysis(
                                    model_config=model_config,
                                    quant_config=quant_config,
                                    dataset_name=args.dataset,
                                    batch_size=args.batch_size,
                                    device_map=args.device_map,
                                    max_new_tokens=args.max_new_tokens,
                                    context_lengths=args.context_lengths,
                                    target_lengths=args.target_lengths,
                                    num_samples=args.num_samples,
                                    baseline_memorized=args.baseline_memorized,
                                    quant_config_swap=quant_config_swap,
                                    layer_swap_config=layer_swap_config,
                                )
                                
                    elif args.swap_every:
                        run_analysis(
                            model_config=model_config,
                            quant_config=quant_config,
                            dataset_name=args.dataset,
                            batch_size=args.batch_size,
                            device_map=args.device_map,
                            max_new_tokens=args.max_new_tokens,
                            context_lengths=args.context_lengths,
                            target_lengths=args.target_lengths,
                            num_samples=args.num_samples,
                            swap_every=args.swap_every,
                            baseline_memorized=args.baseline_memorized,
                            quant_config_swap=quant_config_swap,
                        )
                        
                        
            # if args.layer_swap_config:
            #     layer_swap_configs = load_config_from_json(
            #         Path(args.layer_swap_config), 
            #         config_type="layer_swap"
            #     )
                
            #     for layer_swap_config in layer_swap_configs:
            #         quant_config_swaps = load_config_from_json(
            #             Path(args.quant_config_swap),
            #             config_type="quant"
            #         )
                    
            #         for quant_config_swap in quant_config_swaps:
            #             print(f"Layer Swap Config: {layer_swap_config}")
                        
            #             run_analysis(
            #                 model_config=model_config,
            #                 quant_config=quant_config,
            #                 dataset_name=args.dataset,
            #                 batch_size=args.batch_size,
            #                 device_map=args.device_map,
            #                 max_new_tokens=args.max_new_tokens,
            #                 context_lengths=args.context_lengths,
            #                 target_lengths=args.target_lengths,
            #                 num_samples=args.num_samples,
            #                 baseline_memorized=args.baseline_memorized,
            #                 quant_config_swap=quant_config_swap,
            #                 layer_swap_config=layer_swap_config,
            #             )
            else:
                run_analysis(
                    model_config=model_config,
                    quant_config=quant_config,
                    dataset_name=args.dataset,
                    batch_size=args.batch_size,
                    device_map=args.device_map,
                    max_new_tokens=args.max_new_tokens,
                    context_lengths=args.context_lengths,
                    target_lengths=args.target_lengths,
                    num_samples=args.num_samples,
                    baseline_memorized=args.baseline_memorized,
                )