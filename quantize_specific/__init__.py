__version__ = "0.0.1"
__author__ = 'vision and language group'
__credits__ = 'IITR'

from dataclasses import dataclass
from typing import Optional, Iterator, Dict, List
import json
from pathlib import Path
import argparse
from typing_extensions import Literal
import torch
import numpy as np
import random
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging
)

logging.set_verbosity_error()

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
        model_name: str,
        model_family: str,
        quant_config: QuantConfig,
        quant_config_swap: Optional[QuantConfig],
        layer_swap_config: Optional[LayerSwapConfig],
        swap_every: Optional[List[str]],
        device_map: Literal["cpu", "auto", "balanced"] = "balanced",
        dtype_map: Dict = create_dtype_map(),
    ):
        if layer_swap_config is not None and swap_every is not None:
            raise ValueError(f"Please specify only one of layer_swap_config or swap_every")
        self.dataset = None
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
            model_name,
            padding_side="left",
            use_fast=True,
            clean_up_tokenization_spaces=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = self.quant_config,
            device_map=self.device_map
        )
        self.model.eval()

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

            if quant_config_swap.load_in_4bit:
                self.load_in_swap = "4bit/" +  quant_config_swap.bnb_4bit_quant_type
                if quant_config_swap.bnb_4bit_use_double_quant:
                    self.load_in_swap += "/double"
                else: self.load_in_swap += "/single"
                
            elif quant_config_swap.load_in_8bit:
                self.load_in_swap = "8bit"
                
            self.model_swap = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config = self.quant_config_swap,
                device_map=self.device_map
            )
            self.model_swap.eval()
            self.decoder_map = generate_decoder_map()
            
            try:
                decoders_1 = eval(f"self.model.{DecoderMap[model_family]}")
                decoders_2 = eval(f"self.model_swap.{DecoderMap[model_family]}")
            except AttributeError as e:
                    raise ValueError(f"Unsupported model family '{model_family}' \
                        or invalid layer attribute: {e}")

            if layer_swap_config:
                self.layer_swap_config = layer_swap_config
                self.log_path = (
                    f"./logs/model={model_name}/compute_dtype={self.dtype}/"
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
                    f"./logs/model={model_name}/compute_dtype={self.dtype}/"
                    f"load_in={self.load_in}/"
                    f"quantize_specific/swap_dtype={self.dtype_swap}/"
                    f"load_in_swap={self.load_in_swap}/"
                    f"swap_every={'_'.join(s.replace(' ', '_').replace('/', '%') for s in swap_every)}"
                )
                print(f"log_path: {self.log_path}")
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
