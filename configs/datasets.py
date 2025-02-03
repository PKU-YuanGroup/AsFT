# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"

@dataclass
class pure_bad_dataset:
    dataset: str =  "pure_bad_dataset"
    train_split: str = "ft_datasets/pure_bad_dataset/pure_bad_50.jsonl"


@dataclass
class gsm8k_dataset:
    dataset: str =  "gsm8k_dataset"
    train_split: str = "train"
    test_split: str = "val"

@dataclass
class agnews_dataset:
    dataset: str =  "agnews_dataset"
    train_split: str = "train"
    test_split: str = "val"

@dataclass
class SST2_dataset:
    dataset: str =  "SST2_dataset"
    train_split: str = "train"
    test_split: str = "val"