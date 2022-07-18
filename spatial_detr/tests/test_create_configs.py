CONFIG_DIR_FROZEN_1 = "../../../configs/submission/frozen_1"
CONFIG_DIR_FROZEN_4 = "../../../configs/submission/frozen_4"

import pathlib
import mmcv

def _test_cfg_files(cfg_files):
    for cfg_file in cfg_files:
        # build the config
        cfg = mmcv.Config.fromfile(cfg_file)

def test_configs_frozen_4():

    test_path = pathlib.Path(__file__).resolve()
    cfg_root = test_path.joinpath(CONFIG_DIR_FROZEN_4).resolve()
    assert cfg_root.is_dir(), "configs not found"

    cfg_files = [x for x in pathlib.Path(cfg_root).iterdir() if x.is_file()]

    _test_cfg_files(cfg_files)

def test_configs_frozen_1():
    test_path = pathlib.Path(__file__).resolve()
    cfg_root = test_path.joinpath(CONFIG_DIR_FROZEN_1).resolve()
    print("root =", cfg_root)
    assert cfg_root.is_dir(), "configs not found"

    cfg_files = [x for x in pathlib.Path(cfg_root).iterdir() if x.is_file()]

    _test_cfg_files(cfg_files)