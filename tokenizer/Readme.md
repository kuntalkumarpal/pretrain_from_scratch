
# Tokenizer

* Learnt to adapt the model to learn to tokenize the new data nature and create input vocab
* The Input data should in text format
* Should be trained on the train files
* Lets train on human, dire, debian and bintoo data
* Use human and non-norm rest of the files (Norm-Notnorm won't matter since the varnames would be masked)
* create_file_for_tokenizer.ipynb file prepares data for learning tokenizer(txt) and MLM(json)
* Tokenizer source files present in `/scratch/kkpal/NEW_VAREC/tokenizer/source_tokenizer` dir
* Just have individual text files in a folder and give that folder as input to the tokenizer
* Once tokenizer is learnt move the tokenizer file into the base model directory which will be used throughout the process
* After learning it generates two files (vocab.json(dictionary) and merges.txt)

### Training Tokenizer (50K = Same as RoBERTa 50265)

```
python train_bpe_tokenizer.py \
 --input_path /scratch/kkpal/NEW_VAREC/tokenizer/source_tokenizer \
 --vocab_size 50265 \
 --min_frequency 2 \
 --output_path /scratch/kkpal/NEW_VAREC/tokenizer/bpe50K 
```

### Other Experiments (30K)
* Experiments with 30K Vocabulary - Performance low
* Experiments with 100K Vocabulary - Won't Fit
* Experiments with 60K Vocabulary - Won't Fit
