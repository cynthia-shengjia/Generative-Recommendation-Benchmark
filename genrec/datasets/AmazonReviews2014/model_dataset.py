from torch.utils.data import Dataset
from collections import defaultdict
import pickle

class ModelAmazonReviews2014Dataset(Dataset):
    """
    Load Generative Recommender Dataset
    """
    def __init__(
        self,
        data_interaction_files: str,
        data_text_files: str,
        tokenizer,
        config
    ):
        self.config                 = config
        self.data_interaction_files = data_interaction_files
        self.data_text_files        = data_text_files
        self.tokenizer              = tokenizer

        # Read user history sequences and item text infomations
        self.item_reviews: dict     = self._load_item_reviews()
        self.user_seqs: dict        = self._load_user_seqs()

        # 

 
    def _load_item_reviews(self):
        
        item_reviews = defaultdict(str)
        
        with open(self.data_text_files, 'rb') as f:
            item_titiles_dataframe = pickle.load(f)
        
        for _, row in item_titiles_dataframe.iterrows():
            item_id             =  int(row['ItemID'])
            item_context_info   =  row['Title']
            item_reviews[item_id] = item_context_info
        
        return item_reviews
    
    def _load_user_seqs(self):
        user_seqs = defaultdict(list)

        with optn(self.data_interaction_files, 'rb') as f:
            user_seqs_dataframe = pickle.load(f)
        
        for _, row in user_seqs_dataframe.iterrows():
            user_id     = int(row['UserID'])
            user_seqs   = list(row["ItemID"])
            user_seqs[user_id] = user_seqs

        return user_seqs
        
    
    def _transform_senmatic_ids(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self,batch_users):
        pass