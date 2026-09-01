from datasets import load_dataset, Dataset

num_rows = 1000
train_set_stream = load_dataset('ILSVRC/imagenet-1k', split='train', streaming=True) # IterableDataset
subset_stream = train_set_stream.take(num_rows)
local_dataset = Dataset.from_generator(lambda: iter(subset_stream))
local_dataset.save_to_disk('imagenet1k_1000rows')
