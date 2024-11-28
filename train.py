'''
epoch399 acc=79.6875
'''

import argparse
import os
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from load_datasets import TextDataset, CommandDateset
from RobustLog import AttentionClassification
from test import evaluate


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# def train(model, train_loader, optimizer, device):
#     model.train()
#     total_loss = 0
#     for input_ids, attention_mask, labels in tqdm(train_loader):
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         loss = F.cross_entropy(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     avg_loss = total_loss / len(train_loader)
#     return avg_loss

def main():
    # 初始化
    parser = argparse.ArgumentParser(description="Text Classification Training")
    parser.add_argument('--base_file', type=str, required=True, help="Path to the base training file")
    parser.add_argument('--add_file', type=str, default=None, help="Path to the base training file")
    parser.add_argument('--extra_test_file', type=str, default=None, help="Path to the base training file")
    parser.add_argument('--pretrain_model', type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--max_len', type=int, default=64, help="Max sequence length")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output_dir', type=str, default='model_outputs', help="Output directory for model checkpoints")
    parser.add_argument('--num_class', type=int, default=2, help="")
    parser.add_argument('--attn_dim', type=int, default=64, help='')
    parser.add_argument('--lstm_dim', type=int, default=64, help='')
    parser.add_argument('--emb_dim', type=int, default=20, help='')
    parser.add_argument('--debug', type=bool, default=False, help='')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if not args.debug else 'debug'
    ano_strategy = '_'.join(args.add_file.split('_'))[-3:-1] if args.add_file else 'base'
    output_dir_last = os.path.join(args.output_dir, timestamp + '_' + ano_strategy)
    if not os.path.exists(output_dir_last):
        os.makedirs(output_dir_last)
        
    with open(os.path.join(output_dir_last, 'log.txt'), 'w') as f:
        f.write("Epoch, Accuracy\n")
    
    
    # 准备数据
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    
    train_dataset = CommandDateset(args.base_file, args.add_file, args.extra_test_file,
                                   tokenizer, max_len=args.max_len, seed=args.seed, split='train')
    test_dataset = CommandDateset(args.base_file, args.add_file, args.extra_test_file,
                                  tokenizer, max_len=args.max_len, seed=args.seed, split='test')  # ano_example_test
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 初始化模型
    model = AttentionClassification(args.num_class, args.attn_dim, args.lstm_dim, len(tokenizer), args.emb_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.pretrain_model is not None:
        model.load_state_dict(torch.load(args.pretrain_model))

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_pre = 0.0
    best_rec = 0.0
    best_f1 = 0.0
    # 开始训练
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in tqdm(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_x=input_ids, mask=attention_mask)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            import pdb;pdb.set_trace()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}')

        # 保存模型
        curr_acc, curr_pre, curr_rec, curr_f1 = evaluate(model, test_loader, device)
        if curr_f1 > best_f1:
            print(f'Best model updated with F1 score: {best_f1:.4f}')
            model_to_remove = glob.glob(os.path.join(output_dir_last, 'best*'))
            for file in model_to_remove:
                os.remove(file)
            model_save_path = os.path.join(output_dir_last, f'best_epoch_{epoch}.bin')
            torch.save(model.state_dict(), model_save_path)
        else:
            model_to_remove = glob.glob(os.path.join(output_dir_last, 'last*'))
            for file in model_to_remove:
                os.remove(file)
            model_save_path = os.path.join(output_dir_last, f'last_epoch_{epoch}.bin')
            torch.save(model.state_dict(), model_save_path)
        
        best_acc = max(curr_acc, best_acc)
        best_pre = max(curr_pre, best_pre)
        best_rec = max(curr_rec, best_rec)
        best_f1 = max(curr_f1, best_f1)
        
        with open(os.path.join(output_dir_last, 'log.txt'), 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Current Acc = {curr_acc:.4f}, Best Acc = {best_acc:.4f}\n")
            f.write(f"  Current Precision = {curr_pre:.4f}, Best Precision = {best_pre:.4f}\n")
            f.write(f"  Current Recall = {curr_rec:.4f}, Best Recall = {best_rec:.4f}\n")
            f.write(f"  Current F1 = {curr_f1:.4f}, Best F1 = {best_f1:.4f}\n")
            f.write("-" * 20 + "\n")

if __name__ == '__main__':
    main()
