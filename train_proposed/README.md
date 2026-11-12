# Pretrained Ours Models

We provide a pretrained example of our method for demonstration, along with training results from five runs (seeds 11-15). Use seed13 for the quick-start example.

Each run contains:

- `checkpoints/actor.pt`: trained shared actor weights.
- `config.yaml`: model, environment and safety-filter settings for inference.
- `logs/episodes.csv`: training episode metrics.
- `logs/swanlab/`: original training records.

From the repository root:

```bash
python scripts/evaluate.py --method Ours --checkpoint train_proposed/seed13/checkpoints/actor.pt --formation triangle --path sine
```

Add `--render` for the viewer. Change the seed directory, formation or path to run another example. The shared actor is used together with the DOB-CBF safety filter.

The loadable configuration files were recovered from each run's SwanLab records and converted to the current mixed-training schema. Network, dynamics, reward and safety parameters are preserved; the original records remain in `logs/swanlab/`.
