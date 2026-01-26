import polars as pl
from tqdm import tqdm

def polars_parquet_to_inter(
    parquet_paths,
    output_inter_path,
    start_timestamp=1,
    step=1,
):
    print("Loading parquet files...")

    dfs = []
    user_counter = {}   # 记录每个 user_id 出现了几次

    for p in parquet_paths:
        df = pl.read_parquet(p)

        # Python 层做 suffix（保证全局递增）
        new_user_ids = []
        for u in df["user_id"].to_list():
            if u not in user_counter:
                user_counter[u] = 1
            else:
                user_counter[u] += 1
            new_user_ids.append(f"{u}_{user_counter[u]}")

        df = df.with_columns(pl.Series("user_id", new_user_ids))
        dfs.append(df)

    df = pl.concat(dfs)

    # 只保留 user_id 和 sequence
    df = df.select(["user_id", "sequence"])

    with open(output_inter_path, "w", buffering=1024 * 1024) as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")

        for user, seq in tqdm(df.iter_rows(), total=df.height):
            t = start_timestamp
            for item in seq:
                f.write(f"{user}\t{item}\t{t}\n")
                t += step

    print("Saved to:", output_inter_path)


polars_parquet_to_inter(
    parquet_paths=[
        "/home/j00960957/j00960957/llm4rec_add_general/data/items_meta_seq/Movies_and_TV_cold_user_sequences.parquet",
        "/home/j00960957/j00960957/llm4rec_add_general/data/items_meta_seq/Movies_and_TV_hot_user_sequences.parquet",
    ],
    output_inter_path="rebole_data.inter"
)
