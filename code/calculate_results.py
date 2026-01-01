import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


def _is_valid_file(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def _read_time_series(path: Path) -> np.ndarray:
    """
    Read first column as timestamps (float) from a whitespace-separated log file.
    Returns numpy array of shape (N,).
    """
    txt = path.read_text().strip()
    if not txt:
        return np.array([], dtype=float)
    times: List[float] = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            times.append(float(parts[0]))
        except Exception:
            continue
    return np.array(times, dtype=float)


def _duration_from_times(times: np.ndarray) -> float:
    """
    Robust duration computation.
    - If times are strictly increasing: use last-first.
    - If time jumps backwards: sum of non-negative deltas.
    """
    if times.size < 2:
        return 0.0
    dt = np.diff(times)
    # If monotonic (allow tiny numerical noise), use last-first
    if np.all(dt >= -1e-9):
        return float(times[-1] - times[0])
    # Otherwise, accumulate only forward progress
    return float(np.sum(np.clip(dt, 0.0, None)))


def _pick_best_run(base_dir: Path, ros_dir: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Prefer a "paired" run: control_data + (optional) control_timedata + (optional) issafe
    from the SAME directory, choosing the most recently modified control_data.
    """
    candidates: List[Tuple[float, Path, Optional[Path], Optional[Path]]] = []
    for d in [base_dir, ros_dir]:
        cd = d / "control_data.txt"
        if not _is_valid_file(cd):
            continue
        ct = d / "control_timedata.txt"
        isf = d / "issafe.txt"
        ct = ct if _is_valid_file(ct) else None
        isf = isf if _is_valid_file(isf) else None
        candidates.append((cd.stat().st_mtime, cd, ct, isf))

    if not candidates:
        raise FileNotFoundError(
            "找不到 control_data.txt（工程目录与 ~/.ros 都没有）。请先运行 roslaunch 完成一次轨迹跟踪。"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, control_data, control_timedata, issafe = candidates[0]

    # If timedata missing in the chosen directory, try to find the closest-timestamp timedata elsewhere
    if control_timedata is None:
        td_candidates: List[Tuple[float, Path]] = []
        for d in [base_dir, ros_dir]:
            ct = d / "control_timedata.txt"
            if _is_valid_file(ct):
                td_candidates.append((ct.stat().st_mtime, ct))
        if td_candidates:
            # pick the one closest in mtime to control_data (best chance same run)
            td_candidates.sort(key=lambda x: abs(x[0] - control_data.stat().st_mtime))
            control_timedata = td_candidates[0][1]

    # Same for issafe
    if issafe is None:
        is_candidates: List[Tuple[float, Path]] = []
        for d in [base_dir, ros_dir]:
            isf = d / "issafe.txt"
            if _is_valid_file(isf):
                is_candidates.append((isf.stat().st_mtime, isf))
        if is_candidates:
            is_candidates.sort(key=lambda x: abs(x[0] - control_data.stat().st_mtime))
            issafe = is_candidates[0][1]

    return control_data, control_timedata, issafe


def _resolve_log_paths() -> Tuple[Path, Optional[Path], Optional[Path]]:
    base = Path(__file__).resolve().parent  # .../MRPC-2025-homework/code
    pkg_dir = base / "src/quadrotor_simulator/so3_control/src"
    ros_home = Path.home() / ".ros"
    return _pick_best_run(pkg_dir, ros_home)


def check_additional_file(issafe_path: Optional[Path]) -> int:
    """
    issafe.txt: your existing logic uses 'non-empty => collision happened => return 1'
    Keep consistent with your scoring.
    """
    if issafe_path is None or not _is_valid_file(issafe_path):
        return 0
    try:
        content = issafe_path.read_text().strip()
        return 1 if content else 0
    except Exception:
        return 0


def calculate_rmse_and_more():
    control_data_path, control_timedata_path, issafe_path = _resolve_log_paths()

    des_pos_data = []
    pos_data = []
    time_stamps = []
    length_increments = []

    lines = control_data_path.read_text().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{control_data_path} 数据行数不足（<2），无法评估。")

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        # Expected: t des_x des_y des_z x y z ...
        t = float(parts[0])
        des_pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        pos = np.array([float(parts[4]), float(parts[5]), float(parts[6])], dtype=float)

        time_stamps.append(t)
        des_pos_data.append(des_pos)
        pos_data.append(pos)

        if i > 0:
            prev_parts = lines[i - 1].strip().split()
            if len(prev_parts) >= 7:
                prev = np.array([float(prev_parts[4]), float(prev_parts[5]), float(prev_parts[6])], dtype=float)
                length_increments.append(float(np.linalg.norm(pos - prev)))

    des_pos_array = np.array(des_pos_data, dtype=float)
    pos_array = np.array(pos_data, dtype=float)
    time_stamps_array = np.array(time_stamps, dtype=float)
    length_increments_array = np.array(length_increments, dtype=float)

    if des_pos_array.size == 0 or pos_array.size == 0:
        raise ValueError(f"{control_data_path} 中未解析到有效数据行。")

    if des_pos_array.shape != pos_array.shape:
        raise ValueError("期望位置数据和实际位置数据的形状不一致。")

    # RMSE
    diff = des_pos_array - pos_array
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Total time (CRITICAL FIX):
    # Use timedata internally (last-first), NEVER subtract across two different files.
    if control_timedata_path is not None and _is_valid_file(control_timedata_path):
        t2 = _read_time_series(control_timedata_path)
        total_time = _duration_from_times(t2)
    else:
        total_time = _duration_from_times(time_stamps_array)

    # Total length
    total_length = float(np.sum(length_increments_array)) if length_increments_array.size > 0 else 0.0

    # Collision flag
    additional_score = check_additional_file(issafe_path)

    overall_score = 200.0 * rmse + 0.2 * total_time + 0.2 * total_length + 40.0 * additional_score
    return rmse, total_time, total_length, additional_score, overall_score


if __name__ == "__main__":
    rmse, total_time, total_length, additional_score, overall_score = calculate_rmse_and_more()
    print(f"计算得到的均方根误差（RMSE）值为: {rmse}")
    print(f"轨迹运行总时间为: {total_time}")
    print(f"总轨迹长度为: {total_length}")
    print(f"是否发生了碰撞: {additional_score}")
    print(f"综合评价得分为(综合分数越低越好): {overall_score}")

    result_file_path = Path(__file__).resolve().parent.parent / "solutions/result.txt"
    result_file_path.parent.mkdir(parents=True, exist_ok=True)
    result_file_path.write_text(f"{rmse} {total_time} {total_length} {additional_score} {overall_score}\n")
