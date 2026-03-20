import yaml
from pathlib import Path

cfg_path = Path('configs/config.yaml')
smoke_path = Path('configs/config_smoke_5ep.yaml')
cfg = yaml.safe_load(cfg_path.read_text())

cfg['experiment_name'] = 'smoke_test_5_epochs'
cfg['training']['num_epochs'] = 5
cfg['training']['batch_size'] = 2
cfg['validation']['batch_size'] = 4
cfg['logging']['save_frequency'] = 1
cfg['validation']['eval_frequency'] = 1

cfg['data']['num_frames'] = 8
cfg['data']['frame_size'] = 160

cfg['hardware']['num_workers'] = 2
cfg['hardware']['mixed_precision'] = True

cfg['logging']['checkpoint_dir'] = 'checkpoints/smoke_test'
cfg['paths']['logs'] = 'logs/smoke_test'

smoke_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(smoke_path)
