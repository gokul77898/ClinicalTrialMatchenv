"""
Robust Evaluation Script for Phase 3 ClinicalTrialAgent

Runs 100+ episodes with strict evaluation to provide honest, reproducible metrics.
"""

import statistics
from typing import Dict, List
from src.environment import ClinicalTrialEnv
from src.agents.clinical_trial_agent import ClinicalTrialAgent, random_agent


def run_robust_evaluation(num_episodes: int = 100) -> Dict:
    """
    Run robust evaluation with 100+ episodes.
    
    Args:
        num_episodes: Number of episodes to run
        
    Returns:
        Dict with comprehensive evaluation metrics
    """
    print(f"Running robust evaluation with {num_episodes} episodes...")
    
    # Agent evaluation
    agent_results = []
    agent_failure_modes = {"exclusion": 0, "inclusion": 0, "other": 0}
    
    for i in range(num_episodes):
        # Rotate through tasks
        task_ids = ["single_match", "hidden_exclusion", "ambiguous_match"]
        task_id = task_ids[i % len(task_ids)]
        
        env = ClinicalTrialEnv()
        agent = ClinicalTrialAgent()
        
        # Use seed for reproducibility
        result = agent.run_episode(env, task_id=task_id)
        
        # Analyze failure mode
        if not result["success"]:
            # Simple failure analysis based on reward
            if result["reward"] < -0.8:
                agent_failure_modes["exclusion"] += 1
            elif result["reward"] < -0.3:
                agent_failure_modes["inclusion"] += 1
            else:
                agent_failure_modes["other"] += 1
        
        agent_results.append({
            "episode": i + 1,
            "task_id": task_id,
            "selected_trial": result["selected_trial"],
            "reward": result["reward"],
            "steps": result["steps"],
            "success": result["success"]
        })
    
    # Baseline evaluation
    baseline_results = []
    baseline_failure_modes = {"exclusion": 0, "inclusion": 0, "other": 0}
    
    for i in range(num_episodes):
        task_ids = ["single_match", "hidden_exclusion", "ambiguous_match"]
        task_id = task_ids[i % len(task_ids)]
        
        env = ClinicalTrialEnv()
        result = random_agent(env)
        
        # Analyze failure mode
        if not result["success"]:
            if result["reward"] < -0.8:
                baseline_failure_modes["exclusion"] += 1
            elif result["reward"] < -0.3:
                baseline_failure_modes["inclusion"] += 1
            else:
                baseline_failure_modes["other"] += 1
        
        baseline_results.append({
            "episode": i + 1,
            "task_id": task_id,
            "selected_trial": result["selected_trial"],
            "reward": result["reward"],
            "steps": result["steps"],
            "success": result["success"]
        })
    
    # Calculate metrics
    def calculate_metrics(results: List[Dict]) -> Dict:
        successes = [r["success"] for r in results]
        rewards = [r["reward"] for r in results]
        steps = [r["steps"] for r in results]
        
        return {
            "success_rate": statistics.mean(successes) * 100,
            "avg_reward": statistics.mean(rewards),
            "avg_steps": statistics.mean(steps),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0,
            "results": results
        }
    
    agent_metrics = calculate_metrics(agent_results)
    baseline_metrics = calculate_metrics(baseline_results)
    
    # Calculate improvements
    success_improvement = agent_metrics["success_rate"] - baseline_metrics["success_rate"]
    reward_improvement = agent_metrics["avg_reward"] - baseline_metrics["avg_reward"]
    
    return {
        "agent": agent_metrics,
        "baseline": baseline_metrics,
        "failure_modes": {
            "agent": agent_failure_modes,
            "baseline": baseline_failure_modes
        },
        "improvement": {
            "success_rate": success_improvement,
            "avg_reward": reward_improvement
        },
        "episodes": num_episodes
    }


def print_evaluation_results(results: Dict) -> None:
    """Print comprehensive evaluation results."""
    print("\n" + "="*80)
    print("ROBUST EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nModel Performance:")
    print(f"Random baseline: {results['baseline']['success_rate']:.1f}%")
    print(f"Agent: {results['agent']['success_rate']:.1f}%")
    print(f"Improvement: +{results['improvement']['success_rate']:.1f}%")
    
    print(f"\nDetailed Metrics:")
    print(f"Agent - Success: {results['agent']['success_rate']:.1f}%, "
          f"Reward: {results['agent']['avg_reward']:.3f}, "
          f"Steps: {results['agent']['avg_steps']:.1f}")
    print(f"Baseline - Success: {results['baseline']['success_rate']:.1f}%, "
          f"Reward: {results['baseline']['avg_reward']:.3f}, "
          f"Steps: {results['baseline']['avg_steps']:.1f}")
    
    print(f"\nFailure Modes Analysis:")
    agent_failures = results['failure_modes']['agent']
    baseline_failures = results['failure_modes']['baseline']
    
    total_agent_failures = sum(agent_failures.values())
    total_baseline_failures = sum(baseline_failures.values())
    
    if total_agent_failures > 0:
        print(f"Agent failures ({total_agent_failures}):")
        print(f"  Exclusion: {agent_failures['exclusion']} ({agent_failures['exclusion']/total_agent_failures*100:.1f}%)")
        print(f"  Inclusion: {agent_failures['inclusion']} ({agent_failures['inclusion']/total_agent_failures*100:.1f}%)")
        print(f"  Other: {agent_failures['other']} ({agent_failures['other']/total_agent_failures*100:.1f}%)")
    
    if total_baseline_failures > 0:
        print(f"Baseline failures ({total_baseline_failures}):")
        print(f"  Exclusion: {baseline_failures['exclusion']} ({baseline_failures['exclusion']/total_baseline_failures*100:.1f}%)")
        print(f"  Inclusion: {baseline_failures['inclusion']} ({baseline_failures['inclusion']/total_baseline_failures*100:.1f}%)")
        print(f"  Other: {baseline_failures['other']} ({baseline_failures['other']/total_baseline_failures*100:.1f}%)")
    
    print(f"\nKey Insights:")
    if results['agent']['success_rate'] >= 35:
        print("Agent meets target performance (35-55%)")
    elif results['agent']['success_rate'] >= 25:
        print("Agent approaching target performance")
    else:
        print("Agent below target performance")
    
    if results['improvement']['success_rate'] > 20:
        print("Significant improvement over baseline")
    elif results['improvement']['success_rate'] > 10:
        print("Moderate improvement over baseline")
    else:
        print("Limited improvement over baseline")
    
    # Identify main failure type
    if total_agent_failures > 0:
        main_failure = max(agent_failures.items(), key=lambda x: x[1])[0]
        print(f"Main failure cause: {main_failure}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run robust evaluation
    results = run_robust_evaluation(num_episodes=100)
    print_evaluation_results(results)
