#!/usr/bin/env python
import os
import json
import time
import random
import argparse
from glob import glob
from tqdm.auto import tqdm
from collections import Counter

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    SubsetRandomSampler,
)

# Transformers
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)


###############################################################################
# 0. Parse Command Line Arguments (moved to the beginning)
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 model with custom tokenizer and class-aware sampling"
    )
    # Misc flags
    parser.add_argument(
        "--time_data_loading",
        action="store_true",
        default=False,
        help="Time the data loading for train dataloader (default: False)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable fp16 training (default: False)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of dataloader workers (default: 1)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Train batch size per device (default: 64)",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Logging steps (default: 100)"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save steps (default: 1000)"
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=2, help="Save total limit (default: 2)"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=10000,
        help="Number of classes (default: 10000)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    # sample_first_batch default True; add flag to disable it.
    parser.add_argument(
        "--no_sample_first_batch",
        dest="sample_first_batch",
        action="store_false",
        help="Do not sample only the first batch for each class",
    )
    parser.set_defaults(sample_first_batch=True)
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="class_aware",
        help="Sampling mode, e.g., 'iid' or 'class_aware' (default: class_aware)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size (default: medium)",
    )
    parser.add_argument(
        "--load_checkpoint_dir",
        type=str,
        default=None,
        help="Directory to load checkpoint from. (default: None)",
    )

    parser.add_argument(
        "--no_cuda",
        dest="use_cuda",
        action="store_false",
        help="Do not use CUDA even if available",
    )

    args = parser.parse_args()

    print("Arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")

    return args


###############################################################################
# 1. Custom Tokenizer Logic
###############################################################################


def get_or_create_tokenizer(custom_tokenizer_dir="custom_tokenizer"):
    if not os.path.exists(custom_tokenizer_dir):
        print(
            "No custom tokenizer found. Creating one from base GPT-2 and adding special tokens..."
        )
        base_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        special_tokens = {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>",
            "additional_special_tokens": ["<DOWNWARD>"],
        }
        num_added = base_tokenizer.add_special_tokens(special_tokens)
        print(
            f"Added {num_added} special tokens. New vocab size = {len(base_tokenizer)}"
        )

        # Save this new tokenizer for future runs
        base_tokenizer.save_pretrained(custom_tokenizer_dir)
        print(f"Custom tokenizer saved to {custom_tokenizer_dir}.")
    else:
        print(f"Found existing custom tokenizer at {custom_tokenizer_dir}.")

    # Now load the (possibly newly-created) custom tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(custom_tokenizer_dir)
    print(f"Loaded custom tokenizer with vocab size = {len(tokenizer)}")
    return tokenizer


###############################################################################
# 2. Utility Functions
###############################################################################


def get_tsv_paths(num_classes, sample_first_batch):
    with open("process_p31_p279/class_counts.json", "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    starting_entities = set(list(class_counts.keys())[:num_classes])

    tsv_paths_by_class = {}
    for path in glob("./extracted_paths/*/*.tsv"):
        class_dir = os.path.basename(os.path.dirname(path))
        if class_dir in starting_entities:
            tsv_paths_by_class.setdefault(class_dir, []).append(path)

    tsv_paths = []
    if sample_first_batch:
        for class_dir, paths in tsv_paths_by_class.items():
            batch1_files = [p for p in paths if "batch_1" in os.path.basename(p)]
            if batch1_files:
                tsv_paths.append(batch1_files[0])
            else:
                tsv_paths.append(paths[0])
    else:
        for paths in tsv_paths_by_class.values():
            tsv_paths.extend(paths)

    print(f"Found {len(tsv_paths)} TSV files.")
    return tsv_paths


def load_id2label(num_classes):
    with open(
        f"process_paths/vocab_top_{num_classes}.json", "r", encoding="utf-8"
    ) as f:
        id2label = json.load(f)
    return id2label


###############################################################################
# 3. Datasets
###############################################################################


class EfficientLazyDataset(Dataset):
    """
    Builds an index of (file_idx, byte_offset) for each non-empty line in TSV files.
    __getitem__ seeks directly to that offset, reads one line, and tokenizes it.
    """

    def __init__(self, tsv_paths, id2label, tokenizer, max_length):
        self.tsv_paths = tsv_paths
        self.id2label = id2label
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.index_mapping = []

        print("Building file offset index for lazy loading...")
        for file_idx, path in enumerate(tqdm(tsv_paths, desc="Indexing TSV files")):
            with open(path, "rb") as f:
                offset = f.tell()
                line = f.readline()
                while line:
                    if line.strip():
                        self.index_mapping.append((file_idx, offset))
                    offset = f.tell()
                    line = f.readline()
        print(f"Dataset contains {len(self.index_mapping)} samples.")

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        file_idx, offset = self.index_mapping[idx]
        path = self.tsv_paths[file_idx]

        with open(path, "rb") as f:
            f.seek(offset)
            line = f.readline().decode("utf-8")

        items = line.strip().split("\t")
        tokens = [self.id2label.get(token, token) for token in items]

        if tokens:
            # Build <BOS> tok1 <DOWNWARD> tok2 ... <EOS>
            sequence = self.tokenizer.bos_token + tokens[0]
            for token in tokens[1:]:
                sequence += self.tokenizer.additional_special_tokens[0] + token
            sequence += self.tokenizer.eos_token
        else:
            sequence = self.tokenizer.bos_token + self.tokenizer.eos_token

        encoding = self.tokenizer(
            sequence,
            truncation=False,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = item["input_ids"].clone()
        return item


class SubsetLazyDataset(EfficientLazyDataset):
    def __init__(self, base_dataset, subset_indices):
        self.tsv_paths = base_dataset.tsv_paths
        self.id2label = base_dataset.id2label
        self.tokenizer = base_dataset.tokenizer
        self.max_length = base_dataset.max_length
        self.index_mapping = [base_dataset.index_mapping[i] for i in subset_indices]

    def __len__(self):
        return len(self.index_mapping)


###############################################################################
# 4. Custom Trainer with Optional Class-Aware Sampling
###############################################################################


class MyTrainer(Trainer):

    def __init__(self, sampling_mode="iid", **kwargs):
        self.sampling_mode = sampling_mode
        super().__init__(**kwargs)

    def _get_dataloader_with_sampling(self, dataset, batch_size, shuffle):
        """
        Returns a DataLoader for the given dataset. If class-aware sampling is selected,
        it applies a WeightedRandomSampler; otherwise, it falls back to simple shuffling.
        """
        if self.sampling_mode == "class_aware":
            class_labels = []
            for file_idx, _ in dataset.index_mapping:
                tsv_path = dataset.tsv_paths[file_idx]
                class_label = os.path.basename(os.path.dirname(tsv_path))
                class_labels.append(class_label)
            counts = Counter(class_labels)
            weights = [1.0 / counts[label] for label in class_labels]
            print(f"Computed sample weights for {len(weights)} samples.")

            # Use numpy-based sampling if the dataset is very large
            if len(weights) > 2**24:
                import numpy as np

                weights_np = np.array(weights, dtype=np.float64)
                weights_np /= weights_np.sum()
                indices = np.random.choice(
                    len(weights_np),
                    size=len(dataset),
                    replace=True,
                    p=weights_np,
                )
                sampler = SubsetRandomSampler(indices)
                print("Using numpy-based sampling.")
            else:
                sampler = WeightedRandomSampler(
                    weights, num_samples=len(dataset), replacement=True
                )
                print("Using WeightedRandomSampler.")

            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=self.data_collator,
            )

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return self._get_dataloader_with_sampling(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
        )

    # Removed get_eval_dataloader since we are not using validation.


###############################################################################
# 5. Main Training Logic
###############################################################################


def main():
    args = parse_args()

    # 1. Get or create tokenizer
    tokenizer = get_or_create_tokenizer("custom_tokenizer")

    # 2. Get TSV paths and id2label
    tsv_paths = get_tsv_paths(
        num_classes=args.num_classes, sample_first_batch=args.sample_first_batch
    )
    id2label = load_id2label(num_classes=args.num_classes)

    # 3. Build Dataset (using full dataset for training)
    full_dataset = EfficientLazyDataset(tsv_paths, id2label, tokenizer, args.max_length)
    train_dataset = full_dataset  # No validation split

    # 4. Define model architecture based on chosen model size
    if args.model_size == "small":
        model_arch = {
            "vocab_size": len(tokenizer),
            "n_embd": 128,
            "n_layer": 2,
            "n_head": 2,
            "n_inner": 512,
            "n_positions": args.max_length,
            "attn_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
        }
    elif args.model_size == "medium":
        model_arch = {
            "vocab_size": len(tokenizer),
            "n_embd": 256,
            "n_layer": 4,
            "n_head": 4,
            "n_inner": 1024,  # typically 4 * n_embd
            "n_positions": args.max_length,
            "attn_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
        }
    elif args.model_size == "large":
        model_arch = {
            "vocab_size": len(tokenizer),
            "n_embd": 512,
            "n_layer": 8,
            "n_head": 8,
            "n_inner": 2048,  # typically 4 * n_embd
            "n_positions": args.max_length,
            "attn_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
        }
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    # 5. Create or load model
    if args.load_checkpoint_dir and os.path.exists(args.load_checkpoint_dir):
        print(f"\nLoading model from checkpoint: {args.load_checkpoint_dir}")
        model = GPT2LMHeadModel.from_pretrained(args.load_checkpoint_dir)
        if model.config.vocab_size != len(tokenizer):
            raise ValueError(
                f"Checkpoint vocab_size ({model.config.vocab_size}) does not match "
                f"the current tokenizer length ({len(tokenizer)}). "
                "Make sure to add special tokens and resize the model before training."
            )
    else:
        print("\nTraining from scratch.")
        custom_config = GPT2Config(**model_arch)
        model = GPT2LMHeadModel(custom_config)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    # 6. Move model to device
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")
    print(f"Total model parameters: {model.num_parameters()}")

    # 7. Training arguments (evaluation disabled)
    training_args = TrainingArguments(
        output_dir=f"./model_output_{args.num_classes}/model_size_{args.model_size}/"
        f"sample_first_batch_{args.sample_first_batch}/sampling_mode_{args.sampling_mode}",
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        eval_strategy="no",  # No evaluation
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16 and (device == "cuda"),
        dataloader_num_workers=args.num_workers,
    )

    # 8. Build trainer (only train_dataset is provided)
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        sampling_mode=args.sampling_mode,
    )

    # 9. (Optional) Time data loading
    if args.time_data_loading:
        print("\nTiming the data loading for train dataloader...")
        train_dataloader = trainer.get_train_dataloader()
        start_time = time.time()
        batch_count = 0
        for batch in tqdm(train_dataloader, desc="Iterating over train batches"):
            batch_count += 1
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total batches: {batch_count}")
        print(f"Time to iterate train set: {total_time:.2f} seconds")
        print(f"Avg time per batch: {total_time / batch_count:.4f} seconds\n")

    # 10. Start training
    print(
        f"\nStarting training. Checkpoint: {args.load_checkpoint_dir or 'None (scratch)'}"
    )
    trainer.train(
        resume_from_checkpoint=(
            args.load_checkpoint_dir if args.load_checkpoint_dir else None
        )
    )

    print("\nTraining complete.\n" + "-" * 60)


if __name__ == "__main__":
    main()
