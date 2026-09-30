import subprocess
import sys
import os
import time


def run_evaluation():

    repeat_count = 1

    target_script = 'eval_CMA.py'

    args = [
        '--config', './configs/Retrieval_flickr.yaml',
        '--cuda_id', '0',

        '--model_list', 'ALBEF', 'TCL', 'CLIP_ViT', 'CLIP_CNN',

        '--source_model', 'CLIP_CNN',
        '--albef_ckpt', './checkpoint/albef_flickr.pth',
        '--tcl_ckpt', './checkpoint/tcl_flickr.pth',
        '--original_rank_index_path', './std_eval_idx/flickr30k/'
    ]

    if not os.path.exists(target_script):
        print(f"Error: Target script '{target_script}' does not exist.")
        print(f"Please ensure the script is in the correct directory.")
        return

    python_executable = sys.executable

    command = [python_executable, target_script] + args

    print("=" * 60)
    print(f"Ready to start execution, planned repeats: {repeat_count}")
    print("Full command:")
    print(' '.join(command))
    print("=" * 60)

    for i in range(repeat_count):
        current_run = i + 1
        print(f"\n>>> [Progress] Starting run {current_run}/{repeat_count}...\n")

        try:

            start_time = time.time()

            subprocess.run(command, check=True)

            elapsed = time.time() - start_time
            print(f"\n<<< [Completed] Run {current_run}/{repeat_count} finished (Time: {elapsed:.2f}s).")
            print("-" * 60)

        except FileNotFoundError:
            print(f"Error: Python interpreter not found.")
            break
        except subprocess.CalledProcessError as e:
            print(f"\n!!! [Error] Run {current_run} failed, return code: {e.returncode}")
            print("Stopping subsequent runs due to error.")
            break
        except KeyboardInterrupt:
            print("\nUser forcibly interrupted execution.")
            sys.exit(0)

    print("\nAll tasks finished.")


if __name__ == '__main__':
    run_evaluation()