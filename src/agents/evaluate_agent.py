"""
Phase 3: Agent Evaluation Script

Evaluates the ClinicalTrialAgent against random baseline across multiple episodes.
Computes success rates and average rewards.
"""

from typing import Dict, List
import statistics
from src.environment import ClinicalTrialEnv
from src.agents.clinical_trial_agent import ClinicalTrialAgent, random_agent
from src.tasks import list_tasks


def evaluate_agent(num_episodes: int = 20, task_ids: List[str] = None) -> Dict:
    """
    Evaluate agent performance across multiple episodes.
    
    Args:
        num_episodes: Number of episodes to run
        task_ids: List of task IDs to evaluate (None = all tasks)
        
    Returns:
        Dict with evaluation metrics
    """
    if task_ids is None:
        task_ids = ["single_match", "hidden_exclusion", "ambiguous_match"]
    
    results = []
    
    for episode in range(num_episodes):
        # Rotate through tasks
        task_id = task_ids[episode % len(task_ids)]
        
        env = ClinicalTrialEnv()
        agent = ClinicalTrialAgent()  # Create fresh agent for each episode
        result = agent.run_episode(env, task_id=task_id)
        
        results.append({
            "episode": episode + 1,
            "task_id": task_id,
            "selected_trial": result["selected_trial"],
            "reward": result["reward"],
            "steps": result["steps"],
            "success": result["success"]
        })
    
    # Calculate metrics
    successes = [r["success"] for r in results]
    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]
    
    metrics = {
        "total_episodes": num_episodes,
        "success_rate": statistics.mean(successes) * 100,
        "avg_reward": statistics.mean(rewards),
        "avg_steps": statistics.mean(steps),
        "max_reward": max(rewards),
        "min_reward": min(rewards),
        "results": results
    }
    
    return metrics


def evaluate_baseline(num_episodes: int = 20, task_ids: List[str] = None) -> Dict:
    """
    Evaluate random baseline performance across multiple episodes.
    
    Args:
        num_episodes: Number of episodes to run
        task_ids: List of task IDs to evaluate (None = all tasks)
        
    Returns:
        Dict with evaluation metrics
    """
    if task_ids is None:
        task_ids = ["single_match", "hidden_exclusion", "ambiguous_match"]
    
    results = []
    
    for episode in range(num_episodes):
        # Rotate through tasks
        task_id = task_ids[episode % len(task_ids)]
        
        env = ClinicalTrialEnv()
        result = random_agent(env)
        
        results.append({
            "episode": episode + 1,
            "task_id": task_id,
            "selected_trial": result["selected_trial"],
            "reward": result["reward"],
            "steps": result["steps"],
            "success": result["success"]
        })
    
    # Calculate metrics
    successes = [r["success"] for r in results]
    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]
    
    metrics = {
        "total_episodes": num_episodes,
        "success_rate": statistics.mean(successes) * 100,
        "avg_reward": statistics.mean(rewards),
        "avg_steps": statistics.mean(steps),
        "max_reward": max(rewards),
        "min_reward": min(rewards),
        "results": results
    }
    
    return metrics


def compare_agents(num_episodes: int = 20) -> Dict:
    """
    Compare ClinicalTrialAgent against random baseline.
    
    Args:
        num_episodes: Number of episodes to run
        
    Returns:
        Dict with comparison results
    """
    print("="*80)
    print("PHASE 3: AGENT EVALUATION")
    print("="*80)
    
    # Evaluate our agent
    print("\n1. Evaluating ClinicalTrialAgent...")
    agent_metrics = evaluate_agent(num_episodes)
    
    # Evaluate baseline
    print("2. Evaluating Random Baseline...")
    baseline_metrics = evaluate_baseline(num_episodes)
    
    # Calculate improvement
    success_improvement = agent_metrics["success_rate"] - baseline_metrics["success_rate"]
    reward_improvement = agent_metrics["avg_reward"] - baseline_metrics["avg_reward"]
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nClinicalTrialAgent:")
    print(f"  Success Rate: {agent_metrics['success_rate']:.1f}%")
    print(f"  Avg Reward: {agent_metrics['avg_reward']:.3f}")
    print(f"  Avg Steps: {agent_metrics['avg_steps']:.1f}")
    
    print(f"\nRandom Baseline:")
    print(f"  Success Rate: {baseline_metrics['success_rate']:.1f}%")
    print(f"  Avg Reward: {baseline_metrics['avg_reward']:.3f}")
    print(f"  Avg Steps: {baseline_metrics['avg_steps']:.1f}")
    
    print(f"\nImprovement:")
    print(f"  Success Rate: +{success_improvement:.1f}%")
    print(f"  Avg Reward: +{reward_improvement:.3f}")
    
    # Check if we meet expectations
    print("\n" + "="*80)
    print("EXPECTATIONS CHECK")
    print("="*80)
    
    if baseline_metrics["success_rate"] <= 20:
        print(f"  Random baseline: {baseline_metrics['success_rate']:.1f}% (expected 0-20%)")
    else:
        print(f"  Random baseline: {baseline_metrics['success_rate']:.1f}% (expected 0-20%)")
    
    if 40 <= agent_metrics["success_rate"] <= 70:
        print(f"  ClinicalTrialAgent: {agent_metrics['success_rate']:.1f}% (expected 40-70%)")
    else:
        print(f"  ClinicalTrialAgent: {agent_metrics['success_rate']:.1f}% (expected 40-70%)")
    
    if success_improvement > 0:
        print(f"  Agent beats baseline: YES (+{success_improvement:.1f}%)")
    else:
        print(f"  Agent beats baseline: NO ({success_improvement:.1f}%)")
    
    return {
        "agent": agent_metrics,
        "baseline": baseline_metrics,
        "improvement": {
            "success_rate": success_improvement,
            "avg_reward": reward_improvement
        }
    }


if __name__ == "__main__":
    # Run evaluation
    results = compare_agents(num_episodes=20)
    
    # Print detailed results for first few episodes
    print("\n" + "="*80)
    print("DETAILED EPISODE RESULTS (First 5)")
    print("="*80)
    
    for i, result in enumerate(results["agent"]["results"][:5]):
        print(f"\nEpisode {result['episode']} ({result['task_id']}):")
        print(f"  Selected: {result['selected_trial']}")
        print(f"  Success: {result['success']}")
        print(f"  Reward: {result['reward']:.3f}")
        print(f"  Steps: {result['steps']}")
