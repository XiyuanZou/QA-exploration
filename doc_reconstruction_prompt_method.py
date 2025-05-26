import argparse
from datasets import load_dataset, load_from_disk
from generator import OpenAIGenerator, HFGenerator
from utils import load_prompt, save_results, keep_first_sentence, get_question
from metrics import compute_rougescore, compute_bertscore
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="dataset/wiki/wiki_100", help='Address for input data file, used when dataset is local')
    parser.add_argument('--output_dir', type=str, default="outputs/doc_reconstruction/qwen-2.5-14b/prompt_method_10trials", help='Address for output results file')
    parser.add_argument('--exploration_model_name', default="Qwen/Qwen2.5-14B-Instruct", type=str) #gpt-4o-mini, deepseek-chat
    parser.add_argument('--state_transition_model_name', default="deepseek-chat", type=str)
    parser.add_argument('--content_reconstruction_model_name', default="deepseek-chat", type=str)
    #parser.add_argument('--openai_key', type=str, default="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", help='OpenAI API key if using OpenAI models')
    parser.add_argument('--openai_key', type=str, default="sk-09be7e2e45db41abbbf3a44c27490e87", help='OpenAI API key if using OpenAI models')
    parser.add_argument('--exploration_budget', type=int, default=10, help='how many actions can be taken before performing the final task')
    args = parser.parse_args()
    
    #load data
    dataset = load_from_disk(args.input_dir)
   
    #load models
    if "gpt" in args.exploration_model_name.lower() or "deepseek" in args.exploration_model_name.lower():
        exploration_model = OpenAIGenerator(args.exploration_model_name, args.openai_key)
    else:
        exploration_model = HFGenerator(args.exploration_model_name)
        
    if "gpt" in args.state_transition_model_name.lower() or "deepseek" in args.state_transition_model_name.lower():
        state_transition_model = OpenAIGenerator(args.state_transition_model_name, args.openai_key)
    else:
        state_transition_model = HFGenerator(args.state_transition_model_name)
        
    if "gpt" in args.content_reconstruction_model_name.lower() or "deepseek" in args.content_reconstruction_model_name.lower():
        content_reconstruction_model = OpenAIGenerator(args.content_reconstruction_model_name, args.openai_key)
    else:
        content_reconstruction_model = HFGenerator(args.content_reconstruction_model_name)
    
    #load prompts
    question_generation_prompt_template = load_prompt("prompts/doc_reconstruction/question_generation_prompt.txt")
    answer_provider_prompt_template = load_prompt("prompts/doc_reconstruction/answer_provider_prompt.txt")
    doc_reconstruct_prompt_template = load_prompt("prompts/doc_reconstruction/doc_reconstruct_prompt.txt")
    
    new_samples = []
    total_scores = {
        "rouge_no_exploration": 0.0,
        "rouge_prompt_exploration": 0.0,
        "bert_no_exploration": 0.0,
        "bert_prompt_exploration": 0.0,
    }
    for sample in tqdm(dataset):
        cur_state = sample["partial_obs"]
        new_sample = {"document":sample["document"], "partial_obs":sample["partial_obs"], "explorations":[]}
        
        for i in range(args.exploration_budget):
            question_generation_prompt = question_generation_prompt_template.format(state=cur_state)
            action = exploration_model.generate_answers(input_texts=[question_generation_prompt])[0][0]
            action = get_question(action)
            answer_provider_prompt = answer_provider_prompt_template.format(document=sample["document"], question=action)
            response = state_transition_model.generate_answers(input_texts=[answer_provider_prompt], max_new_tokens=400)[0][0]
            print("Iteration {}: question: {} \n answer: {}".format(i, action, response))
            new_sample["explorations"].append({"iteration": i, "question": action, "answer": response})
            if "unanswerable" in response.lower():
                continue
            cur_state = cur_state + "\n\n" + response
        
        doc_reconstruct_prompt = doc_reconstruct_prompt_template.format(state=cur_state)
        reconstructed_doc_from_exploration = content_reconstruction_model.generate_answers(input_texts=[doc_reconstruct_prompt], max_new_tokens=2000)[0][0]
        new_sample["reconstructed document from exploration"] = reconstructed_doc_from_exploration
        new_sample["final state"] = cur_state
        
        doc_reconstruct_prompt = doc_reconstruct_prompt_template.format(state=sample["partial_obs"])
        reconstructed_doc_from_partial_obs = content_reconstruction_model.generate_answers(input_texts=[doc_reconstruct_prompt], max_new_tokens=2000)[0][0]
        new_sample["reconstructed document from partial obs"] = reconstructed_doc_from_partial_obs
        
        #summarization_prompt = summarization_prompt_template.format(document=cur_state)
        #summary_from_exploration = content_reconstruction_model.generate_answers(input_texts=[summarization_prompt], max_tokens=1000)[0][0]
        #new_sample["summary from exploration"] = summary_from_exploration
        
        #summarization_prompt = summarization_prompt_template.format(document=sample["document"])
        #summary_from_full_doc = content_reconstruction_model.generate_answers(input_texts=[summarization_prompt], max_tokens=1000)[0][0]
        #new_sample["summary from full doc"] = summary_from_full_doc
        
        #summarization_prompt = summarization_prompt_template.format(document=sample["partial_obs"])
        #summary_from_partial_obs = content_reconstruction_model.generate_answers(input_texts=[summarization_prompt], max_tokens=1000)[0][0]
        #new_sample["summary from partial obs"] = summary_from_partial_obs
        
        #eval
        rouge_no_exploration = compute_rougescore(ref=sample["document"], pred=reconstructed_doc_from_partial_obs)
        rouge_prompt_exploration = compute_rougescore(ref=sample["document"], pred=reconstructed_doc_from_exploration)
        bert_no_exploration = compute_bertscore(ref=sample["document"], pred=reconstructed_doc_from_partial_obs)
        bert_prompt_exploration = compute_bertscore(ref=sample["document"], pred=reconstructed_doc_from_exploration)
        
        #rouge_no_exploration = compute_rougescore(ref=summary_from_full_doc, pred=summary_from_partial_obs)
        #rouge_prompt_exploration = compute_rougescore(ref=summary_from_full_doc, pred=summary_from_exploration)
        #bert_no_exploration = compute_bertscore(ref=summary_from_full_doc, pred=summary_from_partial_obs)
        #bert_prompt_exploration = compute_bertscore(ref=summary_from_full_doc, pred=summary_from_exploration)
        
        print("=== Evaluation Scores ===")
        print(f"ROUGE (No Exploration): {rouge_no_exploration}")
        print(f"ROUGE (Prompt-based Exploration): {rouge_prompt_exploration}")
        print(f"BERTScore (No Exploration): {bert_no_exploration}")
        print(f"BERTScore (Prompt-based Exploration): {bert_prompt_exploration}")
        
        new_sample["rouge_no_exploration"] = rouge_no_exploration
        new_sample["rouge_prompt_exploration"] = rouge_prompt_exploration
        new_sample["bert_no_exploration"] = bert_no_exploration
        new_sample["bert_prompt_exploration"] = bert_prompt_exploration
        
        new_samples.append(new_sample)
        
    save_results(new_samples, args.output_dir)
        
if __name__ == '__main__':
    main()