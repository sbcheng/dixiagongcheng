import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute 
import tsfresh
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler  # 朴素贝叶斯使用MinMax缩放
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
#from tsfresh.examples.robot_execution_failures import download_robot_execution_failures,load_robot_execution_failures

'''
# 模拟原始数据：部分时间点缺失
dates = pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-05'])
values = [10, 30, 50]
data = pd.DataFrame({
    'date': dates,
    'value': values
})
'''
yangben_name="1号线训练集+"

def  duiqi(df_jizhan,complete_dates,jizhan_name,jizhan_class):
# 2. 将数据对齐到完整的时间序列索引
    df_jizhan.set_index('时间', inplace=True)  # 将日期列设为索引
    is_duplicate=df_jizhan.index.duplicated(keep=False)
     #= if_double # keep=False 标记所有重复项
    #df_jizhan.set_option('display.max_columns', None)
    #print(is_duplicate)
    df_jizhan=df_jizhan.drop_duplicates()  #去重
    #print(df_jizhan)
    # 先排序，再 groupby + first,去掉有多个时间段的数据，去重。
    df_jizhan = df_jizhan.sort_values("    ").groupby("时间").first().reset_index()
    df_jizhan.set_index('时间', inplace=True)  # 将日期列设为索引

    data_aligned = df_jizhan.reindex(complete_dates)  # 对齐到完整时间索引
    #data_aligned['标签']=data_aligned.loc['2025-04-25 14:00:00']['标签']   #用这个标签填充整个标签
    data_aligned['标签']=jizhan_class
    #data_aligned['    ']=data_aligned.loc['2025-04-25 14:00:00']['    ']
    #print(jizhan_name)
    data_aligned['    ']=jizhan_name
    data_aligned=data_aligned.fillna(0)
    data_aligned=data_aligned.reset_index()
    #print(data_aligned)
    return(data_aligned)

def features(df,y):
    #extract_settings=ComprehensiveFCparameters()
    del df['标签']
    features = extract_features(df, 
        column_id='    ', 
        column_sort='index',
        #column_value='    ',
        #default_fc_parameters=extract_settings,
        impute_function=impute
        )
    #features.index.name='    '
    print(features)
  


    features.to_excel('output3.xlsx',index=True)
    #压缩维度
    '''
    x_imputed=impute(features)
    
    df.set_index('    ',inplace=True)
'''
    y['标签']=y['标签'].astype(bool)
    y.set_index('    ',inplace=True)
    print(y)

    '''
    print(df['标签'].unique())
    features['label']=[0,0,0,1,1,1,1,1]
    #y=pd.Series(y['标签'].values,index=y['    '].values)
    #print(y)
    x_selected=tsfresh.select_features(features,
        features['label'],
        fdr_level=0.01,
        feature_importance='f_classif',
        ml_task='classification'
        )
    print(x_selected)
    '''

    #download_robot_execution_failures()
    #timeseries, y = load_robot_execution_failures()
    #print(y)

    selector = SelectKBest(score_func=f_classif, k=20)  # 先获取所有特征的评分
    x_selected=selector.fit(features,y['标签'].values)

    #X_train, X_test, y_train, y_test = train_test_split(features, y['标签'].values, test_size=0.2, random_state=42)


    #导出
    results = pd.DataFrame({
    'Feature': features.columns,
    'Score': selector.scores_,
    'P-value': selector.pvalues_,
    'Selected': selector.get_support()
    })

    # 按评分降序排序
    results = results.sort_values('Score', ascending=False)
 
# 导出到Excel
    #x_selected.shape.to_excel('123.xlsx',index=False)
    results.to_excel('feature_selection_results.xlsx', index=False)



    
    X_train, X_test, y_train, y_test = train_test_split(features, y['标签'].values, test_size=0.5, random_state=2)

    X_train.to_excel("%s的训练集特征矩阵.xlsx"%yangben_name)  #保存特征矩阵
    joblib.dump(X_train,"%s的训练集特征矩阵.joblib"%yangben_name)

    pipeline = Pipeline([
    ('selector', SelectKBest(score_func=f_classif, k=10)),  # 选择20个最佳特征
    ('scaler', MinMaxScaler(feature_range=(0, 1))),  # 将特征缩放到[0,1]区间
    ('gnb',RandomForestClassifier(random_state=42))# #高斯朴素贝叶斯
    ])
    print("开始训练朴素贝叶斯模型...")
    pipeline.fit(X_train, y_train)

#保存模型
    joblib.dump(pipeline,'gaussianNB小时数据分类7.joblib')
    joblib.dump(X_train,"1号线的训练集特征矩阵7.joblib")


#打印正反例的个数
    print(y['标签'].value_counts())
    # 5. 评估模型
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)#[:, 1]  # 获取正类的预测概率
    #y_pred = (y_proba > 0.99).astype(int)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")


    # 6. 查看被选中的特征
    selected_features = X_train.columns[pipeline.named_steps['selector'].get_support()]
    print("\n被选中的特征:")
    print(selected_features)

    # 找出分类错误的项
    errors = []
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            errors.append((i, y_test[i], y_pred[i],X_test.index[i],y_proba[i]))
    #print(X_test.index)
    # 显示分类错误的项
    print("\n分类错误的项:")
    for index, true_label, pred_label,name_error,proba in errors:
        print(f"索引: {index}, 真实类别: {true_label}, 预测类别: {pred_label},名称：{name_error},概率评分:{proba}")


    # 7. 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()

    # 8. 可视化特征重要性（基于SelectKBest的得分）
    plt.figure(figsize=(10, 6))
    scores = pipeline.named_steps['selector'].scores_[pipeline.named_steps['selector'].get_support()]
    sorted_idx = np.argsort(scores)
    plt.barh(range(len(sorted_idx)), scores[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), selected_features[sorted_idx])
    plt.xlabel('ANOVA F-value分数')
    plt.title('SelectKBest选中的特征重要性')
    plt.tight_layout()
    plt.show()
'''
import pandas as pd
import numpy as np

# 示例数据创建（实际使用时替换为真实数据加载）
data = {
    '时间': pd.date_range('2025-06-09 00:00', '2025-06-09 23:00', freq='H'),
    '    ': ['A']*24,
    '    ': [10, 0, 0, 15, 20, 0, 0, 25, 30, 0, 35, 0, 0, 40, 45, 0, 50, 55, 0, 0, 0, 60, 65, 0],
    '号码': [5, 0, 0, 8, 12, 0, 0, 15, 18, 0, 20, 0, 0, 22, 25, 0, 28, 30, 0, 0, 0, 32, 35, 0]
}
df = pd.DataFrame(data)
'''
def smooth_zero_fill(series, max_zero_len=2):
    """
    平滑填充零值（支持连续1-2个零值）
    :param series: 输入序列
    :param max_zero_len: 最大处理零值长度
    :return: 处理后的序列
    """
    filled = series.copy().astype(float)  # 转为浮点型便于插值
    is_zero = filled == 0
    groups = is_zero.ne(is_zero.shift()).cumsum()
    zero_groups = groups[is_zero].unique()

    for zg in zero_groups:
        mask = groups == zg
        start, end = mask.idxmax(), mask.index[mask].max()
        length = end - start + 1
        
        if length > max_zero_len:
            continue

        # 寻找有效值边界
        prev_valid = filled[:start][filled[:start] != 0].iloc[-1] if not filled[:start].empty else None
        next_valid = filled[end+1:][filled[end+1:] != 0].iloc[0] if not filled[end+1:].empty else None

        # 边界情况处理
        if prev_valid is None and next_valid is None:
            continue  # 全零片段不处理
        elif prev_valid is None:
            filled[mask] = next_valid
        elif next_valid is None:
            filled[mask] = prev_valid
        else:
            # 线性插值计算
            positions = np.linspace(0, 1, length, endpoint=False)
            filled[mask] = prev_valid + (next_valid - prev_valid) * positions

    return filled.round(2)  # 保留两位小数







