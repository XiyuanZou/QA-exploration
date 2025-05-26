import argparse
from datasets import load_dataset, Dataset
import random
from tqdm import tqdm

def main(): 
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--size_of_o', type=int, default=3, help="number of selected paragraphs as partial observation")
    parser.add_argument('--output_dir', type=str, default="dataset/wiki/wiki_100")
    args = parser.parse_args()
    
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    samples = dataset.shuffle(seed=42).select(range(args.num_samples))
    
    total_k_length = 0
    total_o_length = 0
    new_samples = []
    for sample in tqdm(samples):
        text = sample["text"]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 15 or len("".join(paragraphs)) < 1500: #filter out too short doc
            continue 
        
        K = paragraphs[:12] #choose first 10 paragraphs as K
        try:
            selected_o = random.sample(K, args.size_of_o) #choose o from K
        except:
            print(K)
        K = "\n\n".join(K)
        selected_o = "\n\n".join(selected_o)
        total_k_length += len(K)
        total_o_length += len(selected_o)
        new_samples.append({"document": K, "partial_obs": selected_o})
    
    print("Number of samples {}, Average K length {}, Average o length {}".format(len(new_samples), total_k_length/len(new_samples), total_o_length/len(new_samples)))
    new_dataset = Dataset.from_list(new_samples)
    new_dataset.save_to_disk(args.output_dir)

if __name__ == '__main__':
    main()
    
    