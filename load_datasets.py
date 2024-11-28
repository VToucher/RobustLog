import csv
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CommandDateset(Dataset):
    def __init__(self, base_path, add_path, extra_test_path, tokenizer, max_len=64, seed=42, split='train'):
        self.samples = []
        self.max_len = max_len
        self.tokenizer = tokenizer
        add_samples = []
        extra_test_samples = []
        
        # train set来自正常+异常示例的分层抽样+异常补充
        # test set仅来自正常+异常示例的分层抽样
        with open(base_path, 'r', encoding='utf-8') as in_file:
            reader = csv.reader(in_file, delimiter='\u0001')
            header = next(reader)
            idx_behavior = header.index('behavior_cmd')
            idx_scene_id = header.index('scene_id')
            for row in reader:
                cmd = row[idx_behavior].strip()
                label = int(row[idx_scene_id])
                self.samples.append((cmd, label))
        
        if add_path is not None:
            with open(add_path, 'r', encoding='utf-8') as in_file:
                reader = csv.reader(in_file, delimiter='\u0001')
                header = next(reader)
                idx_behavior = header.index('behavior_cmd')
                idx_scene_id = header.index('scene_id')
                for row in reader:
                    cmd = row[idx_behavior].strip()
                    label = int(row[idx_scene_id])
                    add_samples.append((cmd, label))
                    
        if extra_test_path is not None:
            with open(extra_test_path, 'r', encoding='utf-8') as in_file:
                reader = csv.reader(in_file, delimiter='\u0001')
                header = next(reader)
                idx_behavior = header.index('behavior_cmd')
                idx_scene_id = header.index('scene_id')
                for row in reader:
                    cmd = row[idx_behavior].strip()
                    label = int(row[idx_scene_id])
                    extra_test_samples.append((cmd, label))
        
        # 根据标签比例分层抽样
        # train=(1054, 122), test=(264, 30)
        train_set, test_set = train_test_split(self.samples, train_size=0.8, random_state=seed, stratify=[label for _, label in self.samples])
        if split == 'train':
            self.samples = train_set + add_samples  # augmented train set
        elif split == 'ano_example_train':
            self.samples = [item for item in train_set if item[1] == 0]
        elif split == 'ano_example_test':
            self.samples = [item for item in test_set if item[1] == 0]
        elif split == 'test':
            self.samples = test_set
        elif split == 'extra_test':
            self.samples = extra_test_samples
        else:
            raise ValueError
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return input_ids, attention_mask, label

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=64, seed=42, split='train'):
        self.samples = []
        self.max_len = max_len
        self.tokenizer = tokenizer
        
        with open(file_path, 'r', encoding='utf-8') as in_file:
            reader = csv.reader(in_file)
            for row in reader:
                news_desc = row[2].strip()
                news_label = int(row[0]) - 1  # 原数据集是1-4 -> 0-3
                self.samples.append((news_desc, news_label))
        
        train_set, test_set = train_test_split(self.samples, train_size=0.9, random_state=seed)
        if split == 'train':
            self.samples = train_set
        else:
            self.samples = test_set
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return input_ids, attention_mask, label