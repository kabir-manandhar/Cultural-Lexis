import pandas as pd
from collections import Counter
import re
import os
import json
import statistics
import numpy as np
from typing import List, Union, Dict
import argparse
from collections import defaultdict
import json
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon


def earth_movers_distance(arr1: List[int], arr2: List[int]) -> float:
    '''
    Calculate the Earth Mover's distance or Wasserstein 1-distance 
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) 
    between two arrays

    - arr1: Array 1
    - arr2: Array 2

    '''
    return wasserstein_distance(arr1, arr2)



def evaluate_earth_move_scores(gt_answers: List[list], model_scores: List[list]):
    '''
    use this function if you want to input two lists
    '''
    em_scores = []
    for gt_answer, model_score in zip(gt_answers, model_scores): 
        em_score = earth_movers_distance(gt_answer, model_score)
        em_scores.append(em_score)  # Use append instead of extend
    
    mean_em_score = statistics.mean(em_scores)
    print(f"### Overall em_scores: {mean_em_score}")
    return em_scores, mean_em_score



def evaluate_model_distributions(path):
    """Analyze distribution similarity metrics by option count level

    Example use: 
    from evaluate import evaluate_model_distributions
    path = "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_vanilla.json"
    results_tmp = evaluate_model_distributions(path)
    results_tmp

    """
    # Load data
    with open(path, 'r') as f:
        results = json.load(f)
    
    # Filter valid items and calculate metrics
    valid_results = []
    for item in results:
        try:
            if 'choice_values' in item and 'gt_values' in item:
                # Calculate all distribution metrics
                metrics = calculate_distribution_metrics(item['choice_values'], item['gt_values'])
                
                # Count options in survey_scores
                option_num = len(item['survey_scores'].keys()) if 'survey_scores' in item else None
                
                # Store all results with metrics
                valid_results.append({
                    'option_num': option_num,
                    **metrics  # Unpack all metrics into the item
                })
        except Exception as e:
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(valid_results)
    
    # Categorize option numbers
    df['level'] = df['option_num'].apply(lambda x: str(int(x)) if x and x < 5 else '>=5')
    
    # Results by level - create a nested dictionary of metrics
    level_metrics = {}
    metric_names = [col for col in df.columns if col not in ['option_num', 'level']]
    
    # Get levels present in the data
    levels = df['level'].unique().tolist()
    if 'overall' not in levels:
        levels.append('overall')
    
    # Calculate means for each metric by level
    for metric in metric_names:
        level_means = df.groupby('level')[metric].mean().to_dict()
        level_means['overall'] = df[metric].mean()  # Add overall average
        level_metrics[metric] = level_means
    
    # Get counts by level
    counts = df.groupby('level').size().to_dict()
    counts['overall'] = len(df)
    
    return {
        'metrics': level_metrics,
        'counts': counts
    }


def calculate_distribution_metrics(pred, gt):
    """Calculate multiple distribution similarity metrics between prediction and ground truth"""
    # Extract and normalize values
    pred_values = np.array(pred)
    gt_values = np.array(gt)
    
    pred_values /= np.sum(pred_values)
    gt_values /= np.sum(gt_values)
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-10
    positions = np.arange(1, len(gt_values) + 1)
    
    # Calculate all metrics at once
    metrics = {
        'earth_movers_distance': wasserstein_distance(positions, positions, gt_values, pred_values),
        # 'earth_movers_distance': wasserstein_distance( gt_values, pred_values),
        'jensen_shannon': jensenshannon(gt_values, pred_values),
        # 'total_variation': 0.5 * np.sum(np.abs(gt_values - pred_values)),
        # 'hellinger_distance': np.sqrt(0.5 * np.sum((np.sqrt(gt_values) - np.sqrt(pred_values))**2))
        # "argmax": 
    }
    
    # Add normalized EMD
    metrics['earth_movers_distance_normalized'] = metrics['earth_movers_distance'] / (positions[-1] - positions[0])
    
    # Safe versions for KL divergence
    gt_safe = gt_values + epsilon
    pred_safe = pred_values + epsilon
    gt_safe /= np.sum(gt_safe)
    pred_safe /= np.sum(pred_safe)
    
    # KL divergence (both directions)
    metrics['kl_divergence_gt_to_pred'] = entropy(gt_safe, pred_safe)
    metrics['kl_divergence_pred_to_gt'] = entropy(pred_safe, gt_safe)
    metrics['hard_argmax'] = float(np.argmax(pred_safe) == np.argmax(gt_safe))
    
    
    # Cosine similarity
    norm_product = np.linalg.norm(gt_values) * np.linalg.norm(pred_values)
    cosine_sim = np.dot(gt_values, pred_values) / norm_product if norm_product > 0 else 0
    metrics['cosine_similarity'] = cosine_sim
    metrics['cosine_distance'] = 1 - cosine_sim
    
    return metrics



def usage_test():
    print( wasserstein_distance([0, 1, 3], [5, 6, 8]) ) 
    print( wasserstein_distance([0, 1], [0, 1], [3, 1], [2, 2]) ) 
    print( wasserstein_distance([3.4, 3.9, 7.5, 7.8], [4.5, 1.4],
                        [1.4, 0.9, 3.1, 7.2], [3.2, 3.5]))


def get_gt_answers(json_file: str, gt_key: str):
    """Loads questions from the JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    question_to_score = {item['question']: item[gt_key] for item in data}
    return question_to_score



def get_pred_answers(json_file: str, pred_key: str):
    """Loads questions from the JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    question_to_score = {item['question']: item[pred_key] for item in data}
    return question_to_score

def evaluate_earth_move_scores_from_files(args):
    '''
    Use this function if the scores are all stored and you want to post-evaluate them
    '''

    country_name_to_gt_key = {"United States": "us_score", "China": "china_score"}

    gt_answers = get_gt_answers(args.gt_file, gt_key = country_name_to_gt_key[args.country_name] )
    
    pred_answers = get_pred_answers(args.pred_file, pred_key = "normalized_probs" )

    em_scores = defaultdict()
    for question, gt_answer in gt_answers.items():
        model_score = pred_answers[question]
        model_score = list(model_score.values())
        # print(gt_answer)
        # print(model_score)
    
        em_score = earth_movers_distance(gt_answer, model_score)
        em_scores[question] = em_score 
        # print(em_score)

    mean_em_score = statistics.mean(em_scores.values())

    
    for question, em_score in em_scores.items():
        print(question, em_score)

    print(f"### Overall em_scores: {mean_em_score}")



def main():
    parser = argparse.ArgumentParser(description="Run evaluation.")
    parser.add_argument('--gt_file', type=str, default="/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/WV_Bench/us_question_metadata.json", 
                        help='Input JSON file path containing questions')
    parser.add_argument('--pred_file', type=str, default="/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/output/WV_answers.json", 
                        help='Input JSON file path containing model predicted scores')
    parser.add_argument('--country_name', type=str, default='United States', 
                        help='Country context (e.g., "United States" or "China")')

    parser.add_argument('--cache_dir', type=str, default='/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/huggingface_cache', 
                        help='Cache directory')
    
    

    args = parser.parse_args()

    
    
if __name__ == "__main__":
    main()

