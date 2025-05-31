import numpy as np
import os
import shutil
from pathlib import Path

def transform_got10k():
    result_dir = Path('/hy-tmp/CSA/results/GOT-10k/')
    src_dir = result_dir / 'ddpm'
    dest_dir = result_dir / 'got10k_submit'

    # 验证源目录
    if not src_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")
    print(f"源目录包含 {len(list(src_dir.iterdir()))} 个文件")

    # 创建目标目录
    dest_dir.mkdir(exist_ok=True, parents=True)

    for item in src_dir.iterdir():
        if "all" in item.name:
            continue

        # 构造目标路径
        seq_name = item.stem.replace('_time', '')
        seq_dir = dest_dir / seq_name
        seq_dir.mkdir(exist_ok=True)

        try:
            if "time" not in item.name:
                # 处理边界框文件
                dest_file = seq_dir / f"{seq_name}_001.txt"
                bbox_arr = np.loadtxt(item, dtype=int, delimiter='\t')
                np.savetxt(dest_file, bbox_arr, fmt='%d', delimiter=',')
            else:
                # 处理时间文件
                dest_file = seq_dir / item.name
                shutil.copy2(item, dest_file)
        except Exception as e:
            print(f"处理文件失败: {item}")
            print(f"错误详情: {str(e)}")
            continue

    # 压缩前确认
    print(f"即将压缩目录: {src_dir}")
    shutil.make_archive(str(src_dir), "zip", str(src_dir))
    
    # 安全删除
    if input(f"确认删除原始目录 {src_dir}? (y/n)").lower() == 'y':
        shutil.rmtree(src_dir)
    else:
        print("保留原始目录")

if __name__ == "__main__":
    transform_got10k()