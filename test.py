'''
epoch_best
- ano_example_train: Accuracy: 0.91 Precision: 1.00 Recall: 0.91 F1 Score: 0.95
- ano_example_test: Accuracy: 0.89 Precision: 1.00 Recall: 0.89 F1 Score: 0.94
epoch_0
- ano_example_train: 全0
- ano_example_test: 全0
'''

import argparse
import torch
import numpy as np

from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from load_datasets import TextDataset, CommandDateset
from RobustLog import AttentionClassification

def evaluate(model, test_loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_x=input_ids, mask=attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    # import pdb;pdb.set_trace()
    np.save('data/ano_mine_1016/predicted_labels_all_scene.npy', predicted_labels)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    return accuracy, precision, recall, f1

def test_model():
    parser = argparse.ArgumentParser(description="Test Trained Model")
    parser.add_argument('--base_file', type=str, required=True, help="Path to the base training file")
    parser.add_argument('--add_file', type=str, default=None, help="Path to the base training file")
    parser.add_argument('--extra_test_file', type=str, default=None, help="Path to the base training file")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument('--split', type=str, default='test', help="test data split")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--max_len', type=int, default=64, help="Max sequence length")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--num_class', type=int, default=2, help="")
    parser.add_argument('--attn_dim', type=int, default=64, help='')
    parser.add_argument('--lstm_dim', type=int, default=64, help='')
    parser.add_argument('--emb_dim', type=int, default=20, help='')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = CommandDateset(args.base_file, args.add_file, args.extra_test_file,
                                  tokenizer, max_len=args.max_len, seed=args.seed, split=args.split)  # ano_example_train/ano_example_test
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print('num test_dataset:', len(test_dataset))

    model = AttentionClassification(args.num_class, args.attn_dim, args.lstm_dim, len(tokenizer), args.emb_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载训练好的模型
    model.load_state_dict(torch.load(args.checkpoint))
    
    # 进行评估
    accuracy, precision, recall, f1 = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {accuracy:.2f}', f'Precision: {precision:.2f}', f'Recall: {recall:.2f}', f'F1 Score: {f1:.2f}')

if __name__ == '__main__':
    test_model()
