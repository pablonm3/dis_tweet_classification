
# run benchmark
!python run_benchmark.py --no_speed --save_to_csv \
                                --models distilgpt2 \
                                --sequence_lengths 32 128 512 1024 \
                                --batch_sizes 32 \
                                --inference_memory_csv_file benchmarking/required_memory.csv \
                                --env_info_csv_file benchmarking/env.csv >/dev/null 2>&1  # redirect all prints