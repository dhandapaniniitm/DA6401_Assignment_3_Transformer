from datasets import load_dataset
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

## Special tokens required for sequence modeling
SPECIAL_TOKENS = {
    "<unk>": 0,   # Unknown tokens
    "<pad>": 1,   # Padding tokens
    "<sos>": 2,   # Start of the sentence
    "<eos>": 3,   # End of the sentence
}

class Multi30kDataset:
    def __init__(self, split='train',src_vocab=None,tgt_vocab=None):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        ##pass
        ##Loading Multi30k split
        self.dataset = load_dataset("bentrevett/multi30k",split=split)
        
        ##German tokenizer
        self.spacy_de = spacy.load("de_core_news_sm")
        ##English tokenizer
        self.spacy_en = spacy.load("en_core_web_sm")
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        ##Vocabulary must only be built using training set
        if split == "train":
            self.build_vocab()
        ##Converting text into token indices
        self.process_data()

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        ##raise NotImplementedError

        src_sentences = []
        tgt_sentences = []
        
        for item in self.dataset:
            src_sentences.append(self.tokenize_de(item["de"]))
            tgt_sentences.append(self.tokenize_en(item["en"]))
            
        ##source vocabulary
        self.src_vocab = Vocabulary()

        ##Target vocabulary
        self.tgt_vocab = Vocabulary()
        
        ##Building the vocab from the tokenized content
        self.src_vocab.build_vocab(src_sentences)
        self.tgt_vocab.build_vocab(tgt_sentences)
        
    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        ##raise NotImplementedError
        
        self.examples = []

        for item in self.dataset:
            ##Tokenize source and target sentence
            src_tokens = self.tokenize_de(item["de"])
            tgt_tokens = self.tokenize_en(item["en"])

            ##Adding SOS and EOS tokens for source and targets
            src_ids = (
                [self.src_vocab.stoi["<sos>"]]
                + self.src_vocab.numericalize(src_tokens)
                + [self.src_vocab.stoi["<eos>"]]
            )

            tgt_ids = (
                [self.tgt_vocab.stoi["<sos>"]]
                + self.tgt_vocab.numericalize(tgt_tokens)
                + [self.tgt_vocab.stoi["<eos>"]]
            )

            ##append it in tensors
            self.examples.append((torch.tensor(src_ids),torch.tensor(tgt_ids)))
        
    def tokenize_de(self, text):
        """
        Tokenize all the German sentence.
        """
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenize all the english sentence.
        """
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
        
class Vocabulary:
    """
    Handles token to index and index to token mappings.
    """
    def __init__(self):
        ##String to the index dictionary
        self.stoi = SPECIAL_TOKENS.copy()

        ##Index to the string dictionary
        self.itos = {v: k for k, v in self.stoi.items()}

    def build_vocab(self, sentences, min_freq=2):
        """
        Builds vocabulary using training data only.
        """
        ##Counting the token frequencies
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence)

        ##Start indexing after the special tokens
        idx = len(self.stoi)

        for token, freq in counter.items():
        #for token in sorted(counter.keys()):
            ##Ignoring rare words
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1

    def numericalize(self, tokens):
        """
        Converting the token list into integer ids.
        """
        return [self.stoi.get(token,self.stoi["<unk>"]) for token in tokens]

    def __len__(self):
        return len(self.stoi)
    
def collate_fn(batch, pad_idx=1):
    """
    Pads variable-length sequences inside batch and takes care of null case and last few samples in a batch
    """
    src_batch = [x[0] for x in batch]
    tgt_batch = [x[1] for x in batch]

    ##Pad source and target batches
    src_batch = pad_sequence(
        src_batch,
        padding_value=pad_idx,
        batch_first=True
    )
    tgt_batch = pad_sequence(
        tgt_batch,
        padding_value=pad_idx,
        batch_first=True
    )
    return src_batch, tgt_batch