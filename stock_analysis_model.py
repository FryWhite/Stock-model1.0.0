# stock_analysis_model_v2.py
# =======================================
# 金融板块股票数据分析模型 (升级版)
#
# 主要功能：
# 1. 板块联动分析 (基于价格相关性)
# 2. 隐性子板块分类 (支持两种模式切换)
#    a. 模式A: 基于历史价格收益率 (原有逻辑)
#    b. 模式B: 基于多维度技术指标画像 (新增逻辑)
#
# =======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略一些常见的警告，保持输出整洁
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# === 关键配置区 ===
# 【请根据你的实际情况修改以下路径和模式】

# 1. 设置你的CSV文件所在的文件夹路径
#    例如: 'D:/2508stock/stock/stock data/'
DATA_FOLDER_PATH = 'D:/2508stock/stock/stock data/'

# 2. 设置输出的文件夹路径
#    例如: 'D:/2508 stock results/'
OUTPUT_FOLDER_PATH = 'D:/2508 stock results/'

# 3. 选择分析的目标板块或代码模式
#    如果为 '', 则分析文件夹里所有CSV文件。
#    例如：'00000*.SZ' 匹配所有00开头的深市股票
TARGET_STOCK_PATTERN = ''
# === 关键配置区结束 ===


# --- 辅助函数模块 ---

def load_and_prepare_time_series_data(folder_path, pattern=''):
    """
    【功能】加载并合并CSV数据，生成用于时间序列分析（如相关性）的DataFrame。
    【输出】一个长格式的DataFrame，包含所有股票的历史行情数据。
    """
    all_data = []
    print(f"\n--- [辅助模块] 开始加载时间序列数据: '{folder_path}' ---")
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            if pattern and not fnmatch.fnmatch(filename, pattern + '*'):
                continue
            
            file_path = os.path.join(folder_path, filename)
            stock_code = filename.split('.')[0]
            print(f" -> 正在加载: {filename}")
            try:
                df = pd.read_csv(file_path)
                df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                df.set_index('trade_date', inplace=True)
                df['ts_code'] = stock_code
                all_data.append(df)
            except Exception as e:
                print(f"警告: 无法加载文件 {filename}。错误: {e}")
                
    if not all_data:
        print("错误: 在指定路径下没有找到匹配的CSV文件！")
        return None
        
    combined_df = pd.concat(all_data)
    combined_df.sort_index(inplace=True) # 确保按日期排序
    print(f"数据加载完成，共加载 {len(all_data)} 只股票的数据。")
    return combined_df

def prepare_clustering_features_from_indicators(folder_path, pattern=''):
    """
    【功能】从技术指标中提取最新数据，构建用于“画像”式聚类分析的横向特征矩阵。
    【输出】一个标准化的DataFrame，行是股票代码，列是选定的技术指标。
    """
    print("\n--- [辅助模块] 开始准备技术指标特征矩阵 ---")
    
    # 精心挑选的6个核心指标，分别代表价格、趋势、动能、波动性、资金、活跃度
    feature_cols = [
        'pct_chg',                  # 涨跌幅 (短期价格行为)
        'macd_dif_bfq',             # MACD快线 (中期趋势方向)
        'rsi_bfq_6',                # 6日RSI (短期超买超卖动能)
        'atr_bfq',                  # ATR (真实波动幅度)
        'mfi_bfq',                  # 资金流量指标 (价量配合情况)
        'turnover_rate'             # 换手率 (市场活跃度)
    ]
    
    stock_features_dict = {}
    count = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv') and (not pattern or fnmatch.fnmatch(filename, pattern + '*')):
            file_path = os.path.join(folder_path, filename)
            stock_code = filename.split('.')[0]
            df = pd.read_csv(file_path)
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df.sort_values('trade_date', inplace=True)
            
            # 取最新一个交易日的数据
            latest_data = df.iloc[-1] 
            
            feat_vector = {}
            for col in feature_cols:
                if col in latest_data:
                    feat_vector[col] = latest_data[col]
                else:
                    # 如果某个指标缺失（值为NaN），则用该指标在所有股票中的中位数填充，比均值更抗干扰
                    feat_vector[col] = np.nan 
            stock_features_dict[stock_code] = feat_vector
            count += 1
            
    if count == 0:
        print("错误: 没有找到匹配的股票数据！")
        return None

    feature_df = pd.DataFrame.from_dict(stock_features_dict, orient='index')
    
    # --- 数据清洗和标准化 ---
    # 1. 填充缺失值（使用列的中位数）
    feature_df = feature_df.fillna(feature_df.median())
    
    # 2. 特征标准化 (Z-score Normalization)
    # 这是使用不同量纲指标进行聚类前至关重要的一步！
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)
    
    scaled_feature_df = pd.DataFrame(scaled_features, index=feature_df.index, columns=feature_df.columns)
    
    print(f"成功从 {count} 只股票中提取特征，构建了 {scaled_feature_df.shape[0]} x {scaled_feature_df.shape[1]} 的特征矩阵。")
    return scaled_feature_df


# --- 核心分析模块 ---

def perform_correlation_analysis(combined_df, output_path, top_n=10):
    """
    【功能】分析板块内股票的价格联动性。
    【逻辑】计算所有股票两两之间的收益率相关系数，然后找出平均相关性最高的股票。
    【注意】此模块依赖于 load_and_prepare_time_series_data 的输出。
    """
    print("\n--- 模块一：执行板块联动分析 ---")
    
    returns_df = combined_df.pivot_table(index=combined_df.index, columns='ts_code', values='close_hfq').pct_change()
    correlation_matrix = returns_df.corr()
    
    # 计算每只股票与其他股票的平均相关性（排除自相关）
    avg_correlation = correlation_matrix.mean() - np.diag(correlation_matrix)
    
    top_correlated_stocks = avg_correlation.sort_values(ascending=False).head(top_n)
    
    print(f"板块内平均联动性最高的 {top_n} 只股票:")
    print(top_correlated_stocks)
    
    plt.figure(figsize=(12, 7))
    top_correlated_stocks.sort_values(ascending=True).plot(kind='barh', color='skyblue')
    plt.title(f'金融板块股票平均联动性排名 (Top {top_n})')
    plt.xlabel('平均相关系数')
    plt.ylabel('股票代码')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'sector_correlation_ranking.png'))
    print(f"联动性分析结果已保存至: {output_path}/sector_correlation_ranking.png")
    plt.close()
    
    return top_correlated_stocks

def perform_cluster_analysis(time_series_data, output_path, n_clusters=3, use_indicator_features=False):
    """
    【功能】使用聚类分析对股票进行隐性子板块分类。
    【逻辑】根据参数选择，使用两种不同的数据源进行聚类。
    Args:
        time_series_data (pd.DataFrame): 由 load_and_prepare_time_series_data 产生的时间序列数据。
        output_path (str): 输出路径。
        n_clusters (int): 聚类数量。
        use_indicator_features (bool): True=使用新指标，False=使用旧价格收益率。
    """
    print("\n--- 模块二：执行板块子分类分析 ---")
    print(f"[当前模式]: {'使用多维度技术指标' if use_indicator_features else '使用价格收益率'}")

    # === 步骤1: 准备聚类数据 ===
    if use_indicator_features:
        # 场景B: 使用多维度技术指标特征
        clustering_data = prepare_clustering_features_from_indicators(DATA_FOLDER_PATH, TARGET_STOCK_PATTERN)
        if clustering_data is None:
            print("错误: 获取技术指标特征失败，无法执行聚类。")
            return None
        # 数据本身就是标准化的，且行索引是股票代码
        scaled_data = clustering_data.values
        stock_codes = clustering_data.index
        description_suffix = " (基于技术指标画像)"
    else:
        # 场景A: 使用价格收益率
        if time_series_data is None:
            print("错误: 缺少时间序列数据，无法执行此模式聚类。")
            return None
            
        returns_df = time_series_data.pivot_table(index=time_series_data.index, columns='ts_code', values='close_hfq').dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(returns_df.T) # 对股票进行聚类
        stock_codes = returns_df.columns
        description_suffix = " (基于价格联动)"

    # === 步骤2: 使用肘部法则辅助选择K值 (可视化) ===
    print("正在使用肘部法则寻找最佳聚类数...")
    inertias = []
    max_k = min(len(stock_codes), 10) 
    K_range = range(1, max_k)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k' + description_suffix)
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f'elbow_method_for_k{"_indicator" if use_indicator_features else "_returns"}.png'))
    print(f"肘部法则图已保存至: {output_path}/elbow_method_for_k...")
    plt.close()
    
    # === 步骤3: 执行最终聚类 ===
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    cluster_results = pd.Series(cluster_labels, index=stock_codes, name='cluster')
    
    print(f"已完成 {n_clusters} 个聚类的划分。")
    print("各聚类包含的股票:")
    for i in range(n_clusters):
        stocks_in_cluster = cluster_results[cluster_results == i].index.tolist()
        print(f"  - Cluster {i}: {len(stocks_in_cluster)} 只股票")
        print(f"    {', '.join(stocks_in_cluster[:5])}...")
        
    # === 步骤4: 可视化聚类结果 ===
    # 为每个聚类生成一个表格，展示其特征均值，用于解读聚类性质
    print("\n--- 各聚类特征均值解读 ---")
    if use_indicator_features:
        # 如果使用指标聚类，展示特征均值来解释每个聚类的“画像”
        cluster_means = cluster_results.to_frame().join(clustering_data).groupby('cluster').mean()
        print(cluster_means.round(2)) # 保留两位小数，方便阅读

    # 可视化代表股票的走势图 (仅适用于价格收益率模式)
    if not use_indicator_features and time_series_data is not None:
        print("\n正在绘制各聚类样本股票走势图...")
        for i in range(n_clusters):
            cluster_stocks = cluster_results[cluster_results == i].index
            if len(cluster_stocks) > 0:
                sample_stock = cluster_stocks[0]
                plt.figure(figsize=(12, 6))
                time_series_data[time_series_data['ts_code'] == sample_stock]['close_qfq'].plot(title=f'Cluster {i} Sample Stock: {sample_stock}')
                plt.ylabel('前复权收盘价')
                plt.grid(True)
                plt.savefig(os.path.join(output_path, f'cluster_{i}_sample.png'))
                plt.close()
    elif use_indicator_features:
        print("(指标聚类模式) 特征均值表格已在上方输出，可据此解读各聚类性质。")
        # 进阶：可以在这里添加画雷达图的代码，使解读更直观。
    
    return cluster_results


# --- 主程序入口 ---
if __name__ == "__main__":
    # 初始化完整路径
    data_dir = os.path.normpath(DATA_FOLDER_PATH)
    output_dir = os.path.normpath(OUTPUT_FOLDER_PATH)

    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    # --- 选择分析方法 ---
    # 在这里设置 True 或 False，来切换聚类分析的模式！
    # True -> 新方法：多维度技术指标画像
    # False -> 旧方法：价格收益率联动
    USE_INDICATOR_CLUSTERING = True

    # 执行模块一：板块联动分析 (此模块不依赖聚类模式，始终使用价格数据)
    full_time_series_data = load_and_prepare_time_series_data(data_dir, TARGET_STOCK_PATTERN)
    
    if full_time_series_data is not None:
        correlation_results = perform_correlation_analysis(full_time_series_data, output_dir)
        
        # 执行模块二：板块子分类 (根据你的选择来执行)
        print("\n" + "="*50)
        if USE_INDICATOR_CLUSTERING:
            print("!!! 当前运行模式：多维度技术指标聚类 !!!")
        else:
            print("!!! 当前运行模式：价格收益率聚类 !!!")
        print("="*50)
            
        cluster_results = perform_cluster_analysis(
            full_time_series_data, 
            output_dir, 
            n_clusters=3, # 可以修改这个数字来改变聚类数量
            use_indicator_features=USE_INDICATOR_CLUSTERING
        )

        # 保存分析结果到CSV
        if cluster_results is not None and correlation_results is not None:
            correlation_results.to_csv(os.path.join(output_dir, 'top_correlated_stocks.csv'))
            cluster_results.to_csv(os.path.join(output_dir, f'stock_cluster_assignments{"_indicator" if USE_INDICATOR_CLUSTERING else "_returns"}.csv'))
            
            print("\n" + "="*50)
            print("🎉 所有分析模块运行完成！")
            print("="*50)
            print(f"请检查输出文件夹: {output_dir}")
            print("生成的文件包括：")
            print("  - sector_correlation_ranking.png (板块联动排名图)")
            print("  - elbow_method_for_k...png (聚类肘部法则图)")
            if not USE_INDICATOR_CLUSTERING:
                print("  - cluster_0_sample.png ... (各分类样本走势图)")
            print("  - top_correlated_stocks.csv (联动性排名数据)")
            print("  - stock_cluster_assignments...csv (股票分类结果，文件名含模式标识)")
        else:
            print("分析过程中断，未生成最终结果。")
