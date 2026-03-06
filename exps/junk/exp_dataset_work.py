from datasets import load_dataset, Dataset
hf_token = "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"
ds = load_dataset("KbsdJames/Omni-MATH", token=hf_token)  # typically only one split
split_name = list(ds.keys())[0]
print(ds)
print(split_name)
base = ds[split_name]
print(type(base))
solution = base["solution"][0:5]
for ele in solution:
    print(ele, '\n\n')

from datasets import load_dataset

ds = load_dataset("parquet", data_files="./dataset/Omni-MATH_6/dataset_with_solution_sentences.parquet", split="train")
split_name = list(ds.keys())[0]
print(split_name)
base = ds[split_name]
print(base)
