from .polymer import PolymerRegDataset
from ogb.graphproppred import PygGraphPropPredDataset

def get_dataset(args, load_path, load_unlabeled_name="None"):
    if load_unlabeled_name=='None':
        if args.dataset.startswith('plym'):
            return PolymerRegDataset(args.dataset, load_path)
        elif args.dataset.startswith('ogbg'):
            return PygGraphPropPredDataset(args.dataset, load_path)
    else:
        raise ValueError('Unlabeled dataset {} not supported'.format(load_unlabeled_name))
