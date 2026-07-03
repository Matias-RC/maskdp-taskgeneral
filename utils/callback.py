import os
import time
import torch
import wandb

class WandBSnapshotCallback:
    def __init__(self, snapshot_dir="snapshots", save_every=10000, run_name=None, config=None):
        """
        Args:
            snapshot_dir (str): Directory where model snapshots will be stored.
            save_every (int): Frequency (in iterations) to save snapshots.
            run_name (str): Optional name for the WandB run.
            config (dict/OmegaConf): Configuration parameters to log to WandB.
        """
        self.snapshot_dir = snapshot_dir
        self.save_every = save_every
        
        # Ensure snapshot directory exists
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Performance Tracking Variables
        self.start_time = time.time()
        self.cumulative_steps = 0
        
        # Initialize WandB if it hasn't been initialized already
        if wandb.run is None:
            wandb.init(project="MaskDP-A2C", name=run_name, config=config)

    def __call__(self, iteration, losses_data, env_metrics, agent):
        """
        Processes batch metrics, environment performance, tracks training walls,
        and manages checkpoint snapshot saving.
        """
        # Calculate time metrics
        elapsed_time_seconds = time.time() - self.start_time
        elapsed_time_minutes = elapsed_time_seconds / 60.0

        # Accumulate environment steps gathered during this iteration's rollout phase
        # Defaults to a placeholder calculation if 'num_steps' isn't explicitly passed
        self.cumulative_steps += 1

        metrics = {}

        # 1. Aggregate and average the batch/epoch losses from the fine-tuning step
        if losses_data:
            keys = losses_data[0].keys()
            num_batches = len(losses_data)
            
            for key in keys:
                avg_value = sum(batch[key] for batch in losses_data) / num_batches
                metrics[f"train/{key}"] = avg_value

        # 2. Add Environment Telemetry (mean_reward, err_norm)
        if env_metrics:
            for key, val in env_metrics.items():
                if key != "num_steps":  # Avoid duplicating the step metric counter
                    metrics[f"env/{key}"] = val

        # 3. Add Time vs Steps Performance Telemetry
        metrics["perf/elapsed_time_minutes"] = elapsed_time_minutes
        metrics["perf/cumulative_environment_steps"] = self.cumulative_steps

        # Log combined metrics to WandB using cumulative steps as our base X-axis index
        wandb.log(metrics, step=self.cumulative_steps)
        
        # Safe print fallback that won't key-error if total_loss is absent
        primary_loss = metrics.get('train/total_loss', metrics.get('train/actor_loss', 0.0))
        print(f"[Iter {iteration} | Steps {self.cumulative_steps}] Logged to WandB. "
              f"Reward: {metrics.get('env/mean_reward', 0.0):.2f} | Loss: {primary_loss:.4f} | "
              f"Time: {elapsed_time_minutes:.1f}m")

        # 4. Check and execute checkpoint saving
        if iteration % self.save_every == 0:
            snapshot_path = os.path.join(self.snapshot_dir, f"snapshot_iter_{iteration}.pt")
            
            # Defensive check for config attribute
            agent_config = agent.config if hasattr(agent, 'config') else None
            
            payload = {
                "cfg": agent_config,
                "model": agent.backbone.state_dict()
            }
            
            torch.save(payload, snapshot_path)
            print(f"==> Saved model snapshot to: {snapshot_path}")