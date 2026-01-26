import copy
from recbole.quick_start import run_recbole


from recbole.quick_start import run_recbole

parameter_dict = {
    # dataset path config
    'data_path': '/home/j00960957/j00960957/llm4rec_add_general/recbole_eval',  # 自定义数据集路径
    'dataset': 'rebole_data',
    
    # dataset config
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp',
    'load_col': {
        'inter': ['user_id', 'item_id', 'timestamp'],
        'item': ['item_id', 'title', 'brand', 'categories']
    },
    'ITEM_LIST_LENGTH_FIELD': 'item_length',
    'LIST_SUFFIX': '_list',
    'MAX_ITEM_LIST_LENGTH': 50,
    
    'item_attribute': 'title',
    # Training and evaluation config
    'epochs': 500,
    'train_batch_size': 4096,
    'eval_batch_size': 4096,
    'train_neg_sample_args': None,
    'eval_args': {
        'group_by': 'user',
        'order': 'TO',
        'split': {'LS': 'valid_and_test'},
        'mode': 'full'
    },
    'metrics': ['NDCG', 'Hit'],
    'topk': [1, 5, 10],
    'valid_metric': 'Hit@5',
    'metric_decimal_place': 4,
}

run_recbole(model='BERT4Rec', config_dict=parameter_dict) # Caser, BERT4Rec, SASRec, HGN
