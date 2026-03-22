# Adopted from https://github.com/lm-sys/FastChat. Original copyright below:
# Adopted from tatsu-lab@stanford_alpaca. Original copyright below:
# LLaMA model made more memory-efficient via monkey-patching with FlashAttn.

import os
import copy
import json
import logging
import pathlib
import shutil
import warnings
import transformers
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
from torch.utils.data import Dataset
import wandb

# Suppress PyTorch meta parameter warnings during model loading
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

from mobilevlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mobilevlm.train.trainer import VLMTrainer
from mobilevlm import conversation as conversation_lib
from mobilevlm.model.mobilellama import MobileLlamaForCausalLM
from mobilevlm.utils import tokenizer_image_token
from mobilevlm.model.mobilevlm import load_pretrained_model
import random
import numpy as np

local_rank = None  # local rank in distributed training

def seed(seed=0):
    """
    Set random seeds for reproducibility.
    Args:
        seed (int): seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rank0_print(*args):
    """Print only on rank-0 to avoid duplicate output in distributed training."""
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    """
    Model-related argument configuration.
    Covers model path, version, and multimodal settings.
    """
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")  # base language model path
    version: Optional[str] = field(default="v0")  # model version
    freeze_backbone: bool = field(default=False)  # whether to freeze the backbone
    tune_mm_mlp_adapter: bool = field(default=False)  # fine-tune only the multimodal MLP adapter
    vision_tower: Optional[str] = field(default=None)  # vision encoder path
    mm_vision_select_layer: Optional[int] = field(default=-1)   # vision encoder layer to use (-1 = last)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)  # pretrained multimodal MLP adapter path
    mm_projector_type: Optional[str] = field(default='linear')  # multimodal projector type
    mm_use_im_start_end: bool = field(default=False)  # use image start/end tokens
    mm_use_im_patch_token: bool = field(default=True)  # use image patch tokens
    mm_vision_select_feature: Optional[str] = field(default="patch")  # selected visual feature type
    vision_tower_type: Optional[str] = field(default='clip')  # vision tower type (e.g. CLIP)


@dataclass
class DataArguments:
    """
    Data-related argument configuration.
    Covers data path, preprocessing options, and multimodal settings.
    """
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False  # lazy preprocessing to save memory
    is_multimodal: bool = False  # whether the dataset is multimodal
    image_folder: Optional[str] = field(default=None)  # image folder path
    image_aspect_ratio: str = 'square'  # image aspect ratio handling
    image_grid_pinpoints: Optional[str] = field(default=None)  # image grid pinpoints


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Training argument configuration.
    Extends transformers.TrainingArguments with VLM-specific parameters.
    """
    cache_dir: Optional[str] = field(default=None)  # cache directory
    optim: str = field(default="adamw_torch")  # optimizer type
    remove_unused_columns: bool = field(default=False)  # remove unused dataset columns
    freeze_mm_mlp_adapter: bool = field(default=False)  # freeze the multimodal MLP adapter
    mpt_attn_impl: Optional[str] = field(default="triton")  # MPT attention implementation
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    
    # LoRA parameters
    lora_enable: bool = False  # enable LoRA (low-rank adaptation)
    lora_r: int = 64  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha
    lora_dropout: float = 0.05  # LoRA dropout rate
    lora_weight_path: str = ""  # LoRA weight path
    lora_bias: str = "none"  # LoRA bias setting
    
    # Multimodal parameters
    mm_projector_lr: Optional[float] = None  # learning rate for the multimodal projector
    group_by_modality_length: bool = field(default=False)  # group samples by modality length
    
    # Core training parameters
    distill: int = 0  # enable knowledge distillation (0/1)
    task: str = field(default="pretrain")  # training task (pretrain/finetune)
    
    # Knowledge distillation parameters
    distil_type: str = field(default="kl", metadata={"help": "Type of distillation loss: kl, mse, fkl, rkl, tvd, js, adaptive_kl, sfkl, srkl, ctkd, dkd, taid"})
    distil_ratio_type: str = field(default="equal", metadata={"help": "Strategy for weighting multiple distillation losses: equal, uncertainty"})
    
    # TAID (Temperature-Adaptive Instance Distillation) parameters
    taid_t_start: float = field(default=0.4, metadata={"help": "TAID t_start parameter"})
    taid_t_end: float = field(default=1.0, metadata={"help": "TAID t_end parameter"})
    taid_alpha: float = field(default=5e-4, metadata={"help": "TAID alpha parameter"})
    taid_beta: float = field(default=0.99, metadata={"help": "TAID beta parameter"})
    taid_disable_adaptive: bool = field(default=False, metadata={"help": "TAID disable_adaptive parameter"})
    
    # Checkpoint saving parameters
    save_latest: bool = field(default=True, metadata={"help": "Whether to save latest checkpoint link"})
    save_checkpoint_final: bool = field(default=True, metadata={"help": "Whether to save final checkpoint and distillation adapters"})
    save_logit: bool = field(default=False, metadata={"help": "Whether to save student and teacher logits for analysis"})
    logits_save_dir: str = field(default='./saved_logits4', metadata={"help": "Directory to save logits"})
    logits_save_max_batches: int = field(default=100, metadata={"help": "Maximum number of batches to save logits"})

    # Adapter path
    adapter_dir: str = field(default="", metadata={"help": "Directory to save (pretrain) or load (finetune) distillation adapters. Defaults to {output_dir}/adapters/"})

    # WandB parameters
    use_wandb: bool = field(default=False, metadata={"help": "Whether to enable wandb logging"})
    wandb_project: str = field(default="mobilevlm-distillation", metadata={"help": "WandB project name"})

def maybe_zero_3(param, ignore_status=False, name=None):
    """
    Handle parameters under DeepSpeed ZeRO-3.
    In ZeRO-3, parameters may be sharded and must be gathered before use.
    
    Args:
        param: model parameter
        ignore_status: whether to ignore the ZeroParamStatus check
        name: parameter name (for debugging)
    
    Returns:
        gathered parameter tensor
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    """
    Get the state dict for a PEFT model, compatible with ZeRO-3.
    
    Args:
        named_params: named parameter iterator
        bias: bias setting ("none", "all", "lora_only")
    
    Returns:
        PEFT state dict
    """
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        # weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.state_dict().items(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  # {'human': 'USER', 'gpt': 'ASSISTANT'}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "  # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep  # \n
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))  # 2
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        
        # Check for missing image files and filter them out
        self.list_data_dict = self._filter_missing_images(list_data_dict)
        
    def _filter_missing_images(self, list_data_dict):
        """Filter out samples whose image files are missing."""
        filtered_data = []
        missing_images = []
        
        for i, sample in enumerate(list_data_dict):
            if 'image' in sample:
                image_file = sample['image']
                image_folder = self.data_args.image_folder
                image_path = os.path.join(image_folder, image_file)
                
                if not os.path.exists(image_path):
                    missing_images.append(image_path)
                    continue
                    
            filtered_data.append(sample)
        
        if missing_images:
            rank0_print(f"Found {len(missing_images)} missing image files:")
            for missing_img in missing_images[:10]:  # Only show first 10 missing files
                rank0_print(f"  - {missing_img}")
            if len(missing_images) > 10:
                rank0_print(f"  ... and {len(missing_images) - 10} more missing files")
            rank0_print(f"Original dataset size: {len(list_data_dict)}, Filtered size: {len(filtered_data)}")
        else:
            rank0_print("All image files exist, no filtering needed")
            
        return filtered_data

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            # teacher_processor = self.data_args.teacher_image_processor
            image_path = os.path.join(image_folder, image_file)
            
            # Additional safety check
            if not os.path.exists(image_path):
                rank0_print(f"Warning: Image file not found: {image_path}")
                # Return an empty image tensor as replacement
                crop_size = self.data_args.image_processor.crop_size
                image = torch.zeros(3, crop_size['height'], crop_size['width'])
                # Skip image preprocessing, use empty tensor directly
                skip_image_preprocessing = True
            else:
                image = Image.open(image_path).convert('RGB')
                skip_image_preprocessing = False
            if not skip_image_preprocessing:
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    # teacher_image = teacher_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    # teacher_image = teacher_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # If skip_image_preprocessing is True, image is already a tensor with correct shape
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            # data_dict['teacher_image'] = teacher_image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # data_dict['teacher_image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        data_dict['idx'] = self.list_data_dict[i]['id']
        data_dict['overlen']=0
        if data_dict['input_ids'].shape[0]>2048:
            data_dict['overlen']=1
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        idx, input_ids, labels = tuple([instance[key] for instance in instances if instance['overlen']!=1]
                                  for key in ("idx", "input_ids", "labels"))
        if len(input_ids)==0:
            input_ids = [instances[0]['input_ids']]
            labels = [instances[0]['labels']]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id), idx=idx,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances if instance['overlen']!=1]
            if len(images)==0:
                images = [instances[0]['image']]
            # teacher_images = [instance['teacher_image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
                # batch['teacher_images'] = torch.stack(teacher_images)
            else:
                batch['images'] = images
                # batch['teacher_images'] = teacher_images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    seed()
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    # Initialize wandb on rank 0 only, if enabled
    if training_args.use_wandb and (local_rank == 0 or local_rank == -1):
        wandb.init(
            project=training_args.wandb_project,
            name=f"{training_args.task}_{training_args.output_dir.split('/')[-1]}",
            config={
                "task": training_args.task,
                "distill": training_args.distill,
                "distil_type": training_args.distil_type if hasattr(training_args, 'distil_type') else 'none',
                "distil_ratio_type": training_args.distil_ratio_type if hasattr(training_args, 'distil_ratio_type') else 'none',
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "output_dir": training_args.output_dir,
            }
        )
    
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if model_args.model_name_or_path is not None and 'mpt' in model_args.model_name_or_path:
            raise ValueError("")
        else:
            if training_args.local_rank == 0:
                defined_name = 'MobileLlamaForCausalLM'
                ckpt_path = model_args.model_name_or_path
                
            # Handle the case where model_name_or_path is None
            if model_args.model_name_or_path is None or model_args.model_name_or_path.lower() == 'none':
                # Use a base model path to build the model architecture without loading pretrained weights
                base_model_path = "mtgv/MobileLLaMA-1.4B-Chat"
                rank0_print(f"model_name_or_path is None, initializing model architecture from: {base_model_path}")
                
                # Load config and tokenizer from base model
                model = MobileLlamaForCausalLM.from_pretrained(
                    base_model_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
                
                # Re-initialize model weights (random init)
                rank0_print("Re-initializing model weights...")
                model.init_weights()
                
            else:
                # Load student directly to the correct GPU to avoid CPU RAM OOM
                student_device = f"cuda:{training_args.local_rank}" if training_args.local_rank >= 0 else "cuda"
                model = MobileLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
                    device_map={"": student_device},
                    **bnb_model_from_pretrained_args
                )

            if training_args.distill==1:
                teacher_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
                # Load teacher directly to the correct GPU to avoid CPU RAM OOM.
                # Each DeepSpeed rank loads to its own GPU (local_rank).
                teacher_device = f"cuda:{training_args.local_rank}" if training_args.local_rank >= 0 else "cuda"
                teacher = MobileLlamaForCausalLM.from_pretrained(
                    'mtgv/MobileVLM_V2-7B',
                    cache_dir=training_args.cache_dir,
                    torch_dtype=teacher_dtype,
                    device_map={"": teacher_device},
                )
                teacher.config.use_cache = False

    else: # skip
        # Handle non-multimodal model case
        if model_args.model_name_or_path is None or model_args.model_name_or_path.lower() == 'none':
            base_model_path = "facebook/opt-125m"
            rank0_print(f"model_name_or_path is None, initializing model architecture from: {base_model_path}")
            
            model = transformers.LlamaForCausalLM.from_pretrained(
                base_model_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            
            # Re-initialize model weights
            rank0_print("Re-initializing model weights...")
            model.init_weights()
        else:
            model = transformers.LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    model.config.use_cache = False

    if model_args.freeze_backbone: # skip
        model.model.requires_grad_(False)
    if training_args.distill==1:
        teacher.model.requires_grad_(False)
    
    if training_args.bits in [4, 8]: # skip
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing: # skip
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable: # skip
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if model_args.model_name_or_path is not None and 'mpt' in model_args.model_name_or_path: # skip
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        # Handle the case where model_name_or_path is None
        if model_args.model_name_or_path is None or model_args.model_name_or_path.lower() == 'none':
            base_model_path = "mtgv/MobileLLaMA-1.4B-Chat"
            rank0_print(f"Loading tokenizer from base model path: {base_model_path}")
            
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                base_model_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False
            )
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False
            )

    if model_args.version == "v0": # skip
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5": # skip
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else: # skip
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)  
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        # data_args.teacher_image_processor = teacher_image_processor
        data_args.is_multimodal = True
        
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.vision_tower_type = training_args.vision_tower_type = model_args.vision_tower_type

        if not training_args.lora_enable:
            model.requires_grad_(True)

        if model_args.tune_mm_mlp_adapter: # skip
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter: # skip
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]: # skip
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]: # skip
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if training_args.distill == 1:
        trainer = VLMTrainer(model=model, teacher=teacher.eval(), tokenizer=tokenizer, args=training_args, **data_module)
    else:
        trainer = VLMTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    if training_args.task == 'finetune' and training_args.distill == 1:
        adapter_dir = training_args.adapter_dir if training_args.adapter_dir else os.path.join(training_args.output_dir, 'adapters')
        proj_adapter_path = os.path.join(adapter_dir, 'proj_adapter.pt')
        attn_adapter_path = os.path.join(adapter_dir, 'attn_adapter.pt')
        rank0_print(f"Loading distillation adapters from: {adapter_dir}")
        proj_ckpt = torch.load(proj_adapter_path, map_location='cpu')
        if isinstance(proj_ckpt, dict):
            trainer.proj_adapter.load_state_dict(proj_ckpt)
        else:
            trainer.proj_adapter = proj_ckpt
        trainer.proj_adapter = trainer.proj_adapter.to(dtype=trainer.teacher.dtype, device=trainer.teacher.device)

        attn_ckpt = torch.load(attn_adapter_path, map_location='cpu')
        if isinstance(attn_ckpt, dict):
            trainer.attn_adapter.load_state_dict(attn_ckpt)
        else:
            trainer.attn_adapter = attn_ckpt
        trainer.attn_adapter = trainer.attn_adapter.to(dtype=trainer.teacher.dtype, device=trainer.teacher.device)
    # Start training
    trainer.train()
    
    # Post-training processing
    rank0_print("Training complete. Saving final model...")
    trainer.save_state()

    model.config.use_cache = True

    # Save the full model weights for inference.
    # safe_save_model_for_hf_trainer() is called on all ranks so DeepSpeed collective
    # operations complete correctly (no rank-0-only deadlock risk).
    if training_args.lora_enable:
        rank0_print("Saving LoRA files...")
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            rank0_print('Saving non-LoRA trainable parameters...', list(non_lora_state_dict.keys())[:5])
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            rank0_print("LoRA files saved.")
    else:
        rank0_print("Saving full model...")
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        rank0_print("Full model saved.")
    
    # Print training summary
    rank0_print("="*60)
    rank0_print("Training pipeline complete!")
    rank0_print(f"Output dir: {training_args.output_dir}")
    rank0_print(f"Final model: {training_args.output_dir}")
    if training_args.distill == 1:
        adapter_dir = training_args.adapter_dir if training_args.adapter_dir else os.path.join(training_args.output_dir, 'adapters')
        rank0_print(f"Distillation adapters: {adapter_dir}")
    if training_args.lora_enable:
        rank0_print("LoRA weights saved.")
    rank0_print("="*60)
    
    if training_args.use_wandb and (local_rank == 0 or local_rank == -1):
        wandb.finish()

if __name__ == "__main__":
    train()
