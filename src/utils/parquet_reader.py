import pandas as pd

# 1. ファイルパスを定義
file_path = 'data/analysis/refactoring_instances/commits_with_refactoring.parquet'

try:
    # 2. 必要な列だけを読み込む
    df = pd.read_parquet(file_path, columns=['sha', 'agent', 'title', 'has_refactoring'])
    
    # 3. 条件を指定して行をフィルタリング
    # agentに'cursor'が含まれ、titleに'refactor'が含まれるコミットを抽出
    extracted_data = df[
        (df['agent'].str.contains('cursor', na=False)) & 
        (df['title'].str.contains('refactor', na=False, case=False))
    ]
    
    # 4. 結果を表示
    print(f"抽出されたデータ件数: {len(extracted_data)}")
    print("--- データサンプル ---")
    print(extracted_data.head())

except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません: {file_path}")
except Exception as e:
    print(f"エラーが発生しました: {e}")
