import torch
import dgl
from utils import load_data, EarlyStopping, setup
from sklearn.metrics import f1_score
import argparse
from model import HGIN
import os
import sys
import numpy as np

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        embeds, logits = model(g)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])
    
    return loss, accuracy, micro_f1, macro_f1, embeds

def main(args):
    G, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask = load_data(args['dataset'], ratio=args['ratio'], remove_self_loop=False)
    target_node_type = 'paper'
    meta_paths=[['written-by','writing'],['is-about','has']]
    
    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    model = HGIN(G, features, target_node_type, meta_paths, in_size=features.shape[1], hidden_size=args['hidden_size'], 
                     out_size=num_classes, num_layers=args['num_layers'], num_mlp_layers=args['num_mlp_layers'], dropout=args['dropout'], device=args['device']).to(args['device'])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    stopper = EarlyStopping(args['dataset'], args['patience'])

    for epoch in range(args['num_epochs']):
        model.train()
        _, logits = model(G)
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1, _ = evaluate(model, G, features, labels, val_mask, loss_fn)
        early_stop = stopper.step(val_loss.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} |'
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(epoch+1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)

    # print test results.
    print('**********Evaluate TestSet***********')
    test_loss, test_acc, test_micro_f1, test_macro_f1, embeds = evaluate(model, G, features, labels, test_mask, loss_fn)
    print('Test Loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(test_loss.item(), test_micro_f1, test_macro_f1))
        
    embeds = embeds.cpu().detach().numpy()
    savepath = os.path.join(sys.path[0],'Embedding/EMP_{}%.npy'.format(args['ratio']))  # shape: 
    np.save(savepath, embeds)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-r', '--ratio', type=int, default=20, help='split radio')
    parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed')
    parser.add_argument('-ld','--log-dir',type=str,default='results', help='Dir for saving training results')


    args = parser.parse_args().__dict__
    args = setup(args)
            
    main(args)