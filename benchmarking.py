!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/benchmarking/run_benchmark.py -qq

# run benchmark
python run_benchmark.py --no_speed --save_to_csv \
                                --models distilgpt2 \
                                --sequence_lengths 32 128 512 1024 \
                                --batch_sizes 32 1028\
                                --inference_memory_csv_file benchmarking/required_memory.csv \
                                --env_info_csv_file benchmarking/env.csv




export TRAIN_MEMORY_CSV_FILE="benchmarking/train_memory.csv"
export INFERENCE_MEMORY_CSV_FILE="benchmarking/inference_memory.csv"
python run_benchmark.py --no_speed --save_to_csv \
                                --models distilgpt2 \
                                --sequence_lengths 32 128 512 1024 \
                                --batch_sizes 32 1028\
                                --training
                                --inference_memory_csv_file INFERENCE_MEMORY_CSV_FILE \
                                --env_info_csv_file benchmarking/env.csv \
                                --train_memory_csv_file TRAIN_MEMORY_CSV_FILE


