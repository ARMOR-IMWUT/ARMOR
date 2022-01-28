import torch

from src.datasets.dataset_utils import get_dataset
from src.models.models import get_model
from src.parser import Arguments

import numpy as np

from src.train.train import LocalUpdate

if __name__ == '__main__':
    args = Arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if args.gpu else 'cpu'
    ## load the dataset
    train_dataset, test_dataset, user_groups = get_dataset(args)

    ## load the model

    global_model = get_model(args)
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()

    for epoch in range(args.epochs):
        print(f'\n | Global Training Round : {epoch + 1} |\n')
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx],
                                      test_dataset=test_dataset, logger='')
            print('------------------------------------------')
            print(f'--------------User: {idx}-----------------')
            print('------------------------------------------')

        if args.attack == 1:
            attack = (idx in args.attackers_list) and (epoch > args.start_round)
        elif args.attack == 2:
            attack = (idx in args.attackers_list) and (epoch % args.attack_step == 0) and (
                    epoch > args.start_round)
        elif args.attack == 3:
            pass
        elif args.attack == 4:
            pass
        else:
            attack = False
