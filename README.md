# PythonProject_test

Robot arm + camera pick/place workspace.

## Structure
- `apps/`: actual executable scripts (maintained code)
- `data/calib/`: canonical calibration assets
- `archive/`: notes and historical assets
- `weights/`, `yolov5/`: model/runtime dependencies

## Script policy
All executable scripts are managed under `apps/`.

### Active scripts (recommended to edit)
- `apps/affine_update.py`
- `apps/pick_and_place.py`
- `apps/robot_setup.py`
- `apps/camera_calibration.py`
- `apps/board_pose_calibration.py`
- `apps/experiments/movecali_yolo.py` (legacy experiment)

## Calibration files
- `data/calib/camera_intrinsics_1280x720.npz`: canonical camera intrinsics
- `data/calib/affine_fb.npz`: canonical affine correction
- `data/calib/history/affine/`: affine backup history (auto, on save)
- `data/calib/history/intrinsics/`: intrinsics snapshots

## Typical workflow
1. Camera intrinsics calibration
   - `python apps/camera_calibration.py`
2. Affine update
   - `python apps/affine_update.py`
   - capture `1~9` -> solve `g` -> save `s`
3. Pick-and-place runtime
   - `python apps/pick_and_place.py`
