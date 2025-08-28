# 导入模块
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import sys
import io
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
from tkinter import filedialog
import threading
from ditie.dixiagongchengT import duiqi,features,smooth_zero_fill
from ditie.dixiagongchengP import duiqi2,features2,smooth_zero_fill2,resource_filter,excel_fenleijieguo,features_check


class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget.tag_config("stdout", foreground="#00FF00", background="#000000")  # 荧光绿字 + 黑色背景
 
    def write(self, message):
        self.text_widget.insert("end", message, "stdout")  # 使用 "stdout" 标签应用样式
        self.text_widget.see("end")  # 自动滚动到底部
 
    def flush(self):
        pass  # 必须实现，但可以留空


yangben_name="1号线训练集+"
jizhan_to_remove=[]


def run():
    print('训练脚本，运行成功！')
    try:
        # 读取Excel文件
        df = pd.read_excel(
            "%s.xlsx"%yangben_name,
            sheet_name='Sheet1',
            header=0,
            usecols='A:G',
            #dtype={'Date': str, 'Product': str, 'Quantity': int, 'Price': float}
        )
        
        # 数据预处理
        #df['Total'] = df['Quantity'] * df['Price']
        
        # 显示处理后的数据
        print(df.dtypes)
      
    except FileNotFoundError:
        print("错误：文件未找到，请检查文件路径")
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
    print(df)
    df['时间']=pd.to_datetime(df['时间'])
    print(df.dtypes)
    print(df)

    # 1. 生成完整的时间序列索引
    start_date = '2025-04-21 00:'
    end_date = '2025-04-26 00:'
    complete_dates = pd.date_range(start=df["时间"].min(), end=df["时间"].max(), freq='h')  # 按天生成完整索引

    print(complete_dates)

    total_pd=pd.DataFrame()
    df=df.dropna(subset=[' '])  #删除为空的


    df1=pd.DataFrame()

    #循环 每个按照时间对其
    for jizhan_name in df[' '].unique():
        df_jizhan=df[df[' ']==jizhan_name]
        jizhan_class=df_jizhan['标签'].unique()[0]       #把标签提取出来，稍后使用，注意标签不能有两个
        print(jizhan_class)
        print(jizhan_name)
        #print(len(df[' '].unique()))
        #print(df_jizhan)
        #print(duiqi(df_jizhan,complete_dates))
        df0=duiqi(df_jizhan,complete_dates,jizhan_name,jizhan_class)
        df1=pd.concat([df1,df0],ignore_index=True)
        '''

        plt.plot(df_jizhan.index,df_jizhan['号码'],df_jizhan['信令'])
        plt.legend()
        plt.grid(True)
        plt.title(df_jizhan['标签'][0])
        plt.show()
        
        plot_acf(df['信令'], lags=48,alpha=0.05)
        plt.legend()
        plt.grid(True)
        plt.title(df_jizhan['标签'][0])
        plt.show()
    '''
    print(df1)

    #构造新的一列
    
  # 统计每个名称'数量'为0的单元数量
    zero_counts = df1[df1['信令'] == 0].groupby(' ').size()
    names_to_drop = zero_counts[zero_counts > 150].index   #过滤出僵尸 
    df1 = df1[~df1[' '].isin(names_to_drop)]     #过滤僵尸基
    print(zero_counts)
    print(df1)
    df1.to_excel('%s补充时间后的中间库.xlsx'%yangben_name, index=False)

    #有部分时间段突然为0，使用平滑的方法补值
    # 处理'信令'和'号码'字段
    # 按 分组处理
    df1= df1.groupby(' ').apply(
        lambda x: x.assign(
            信令=smooth_zero_fill(x['信令']),
            号码=smooth_zero_fill(x['号码'])
        )
    ).reset_index(drop=True)
    # 验证结果
    #print("原始数据示例：")
    #print(df[['时间', '信令', '号码']].head(10))
    #print("\n处理后数据示例：")
    #print(processed_df[['时间', '信令', '号码']].head(10))

    # 输出处理后的数据
    print(df1)



   #df['比值']=np.where(df1['号码'] == 0, 0, df1['号码'] / df1['信令'])
    features(df1,df1[[' ','标签']].drop_duplicates())
    #df123=df1[[' ','标签']].drop_duplicates()
    #print(df123)
    stop_event.set()

def run2():
    print('预测脚本，运行成功！')
    bianli_list=[
     ("样本3+","1号线"),
     ("样本3.2+","2号线"),
     ("样本3.4+","4号线"),
    ("样本3.5+","5号线"),
    ("样本3.6+","6号线"),
    ("样本3.9+","9号线"),
    ("样本3.10+","10号线"),
    ("样本3.18+","18号线"),
    ("样本3.19+","19号线"),
    ("样本3.27+","27号线"),
    ("样本3.s3+","s3号线"),
    ("样本3.7+","7号线")
    ]

    tel_companys=["4-","3-","2-","1-"]
    for (yangben_name,number_of_ditie) in bianli_list:    #遍历地铁线
        for tel_company in tel_companys:                  #遍历运营商
            try:
                # 读取Excel文件
                '''
                df = pd.read_excel(
                    "样本2.1+.xlsx",
                    sheet_name='Sheet1',
                    header=0,
                    usecols='D:J',
                    #dtype={'Date': str, 'Product': str, 'Quantity': int, 'Price': float}
                )
                '''
                df=pd.read_csv("%s.csv"%yangben_name)

                # 数据预处理
                #df['Total'] = df['Quantity'] * df['Price']
                df=resource_filter(df,tel_company,jizhan_to_remove)
                
                
                # 显示处理后的数据
                print(df.dtypes)




                
            except FileNotFoundError:
                print("错误：文件未找到，请检查文件路径")
            except Exception as e:
                print(f"读取Excel文件时出错: {e}")
            print(df)
            df['时间']=pd.to_datetime(df['时间'])
            print(df.dtypes)
            print(df)



            # 1. 生成完整的时间序列索引
            start_date = '2025-04-20 00:'
            end_date = '2025-04-27 00:'
            complete_dates = pd.date_range(start=df["时间"].min(), end=df["时间"].max(), freq='h')  # 按天生成完整索引

            print(complete_dates)

            total_pd=pd.DataFrame()
            df=df.dropna(subset=[' '])  #删除为空的


            df1=pd.DataFrame()

            #循环 每个按照时间对其
            for jizhan_name in df[' '].unique():
                df_jizhan=df[df[' ']==jizhan_name]
                jizhan_class=df_jizhan['标签'].unique()[0]       #把标签提取出来，稍后使用，注意标签不能有两个
                #print(jizhan_class)
                #print(jizhan_name)
                #print(len(df[' '].unique()))
                #print(df_jizhan)
                #print(duiqi(df_jizhan,complete_dates))
                df0=duiqi2(df_jizhan,complete_dates,jizhan_name,jizhan_class)
                df1=pd.concat([df1,df0],ignore_index=True)
                '''

                plt.plot(df_jizhan.index,df_jizhan['号码'],df_jizhan['信令'])
                plt.legend()
                plt.grid(True)
                plt.title(df_jizhan['标签'][0])
                plt.show()
                
                plot_acf(df['信令'], lags=48,alpha=0.05)
                plt.legend()
                plt.grid(True)
                plt.title(df_jizhan['标签'][0])
                plt.show()
            '''
            print(df1)

            #构造新的一列
            
          # 统计每个名称'数量'为0的单元数量
            zero_counts = df1[df1['信令'] == 0].groupby(' ').size()
            names_to_drop = zero_counts[zero_counts > 150].index   #过滤出僵尸 

            pd.Series(names_to_drop).to_csv("%s剔除的 .csv"%yangben_name,index=False)
            df1 = df1[~df1[' '].isin(names_to_drop)]     #过滤僵尸基
            #print(zero_counts)
            #print(df1)
            df1.to_csv('%s补充时间后的中间库.csv'%yangben_name, index=False)
            #cc=input("input")

            #有部分时间段突然为0，使用平滑的方法补值
            # 处理'信令'和'号码'字段
            # 按 分组处理
            df1= df1.groupby(' ').apply(
                lambda x: x.assign(
                    信令=smooth_zero_fill2(x['信令']),
                    号码=smooth_zero_fill2(x['号码'])
                )
            ).reset_index(drop=True)
            # 验证结果
            #print("原始数据示例：")
            #print(df[['时间', '信令', '号码']].head(10))
            #print("\n处理后数据示例：")
            #print(processed_df[['时间', '信令', '号码']].head(10))

            # 输出处理后的数据
            print(df1)



           #df['比值']=np.where(df1['号码'] == 0, 0, df1['号码'] / df1['信令'])
            features2(df1,df1[[' ','标签']].drop_duplicates(),yangben_name,number_of_ditie,tel_company)
            #df123=df1[[' ','标签']].drop_duplicates()
            #print(df123)
            stop_event.set()


#多线程优化
def start_task():
    """启动后台任务"""
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

def start_task2():
    """启动后台任务"""
    thread = threading.Thread(target=run2, daemon=True)
    thread.start()

#文件选择器
def select_file(i):
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(title="选择文件")
    if file_path:  # 如果用户选择了文件
        if i==1:
            entry_var1.set(file_path)  # 将路径设置到Entry中
        elif i==2:
            entry_var2.set(file_path)  # 将路径设置到Entry中
        elif i==3:
            entry_var3.set(file_path)  # 将路径设置到Entry中
        elif i==4:
            entry_var4.set(file_path)  # 将路径设置到Entry中






#主程序
# 创建窗口'ts'及其设置
ts = tk.Tk()
ts.title('时间序列分类工具')
ts.geometry('1200x800')
ts.resizable(1, 1)
ts.attributes('-toolwindow', 0)
ts.config(bg='#A8A8A8', cursor='arrow')
ts.overrideredirect(None)
#radiobutton变量初始化
var = tk.StringVar(value=None)  # 初始无选中
var.set(None)
var1 = tk.StringVar(value=None)  # 初始无选中
var1.set(None)
var2 = tk.StringVar(value=None)  # 初始无选中
var2.set(None)
entry_var1 = tk.StringVar()
entry_var2 = tk.StringVar()
entry_var3 = tk.StringVar()
entry_var4 = tk.StringVar()


# 创建组件'Label_1'及其设置
Label_1 = tk.Label(ts, text='源数据路径', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('方正小标宋简体', 9, ), cursor='arrow')
Label_1.place(x=88, y=78, width=80, height=30)
# 创建组件'Entry_1'及其设置
Entry_1 = tk.Entry(ts, text='', textvariable=entry_var1,fg='black', bg='white', state='normal', relief='sunken', bd=1, font=('微软雅黑', 9), cursor='xterm')
Entry_1.place(x=189, y=78, width=412, height=30)
# 创建组件'Label_2'及其设置
Label_2 = tk.Label(ts, text='源模型路径', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('方正小标宋简体', 9, ), cursor='arrow')
Label_2.place(x=88, y=135, width=80, height=30)
# 创建组件'Label_3'及其设置
Label_3 = tk.Label(ts, text='数据预处理', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('方正小标宋简体', 9, ), cursor='arrow')
Label_3.place(x=88, y=199, width=80, height=30)
# 创建组件'Entry_2'及其设置
Entry_2 = tk.Entry(ts, text='', textvariable=entry_var2,fg='black', bg='white', state='normal', relief='sunken', bd=1, font=('微软雅黑', 9), cursor='xterm')
Entry_2.place(x=189, y=135, width=412, height=30)
# 创建组件'Message_1'及其设置
Message_1 = ScrolledText(ts, width=491, height=585, state='normal',bg="#000000", fg="#00FF00")
Message_1.place(x=664, y=78, width=491, height=585)
# 创建组件'Checkbutton_1'及其设置
Checkbutton_1 = tk.Checkbutton(ts, text='稀疏性处理', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('微软雅黑', 9), cursor='arrow')
Checkbutton_1.place(x=189, y=199, width=110, height=30)
# 创建组件'Label_4'及其设置
Label_4 = tk.Label(ts, text='阈值', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('微软雅黑', 9), cursor='arrow')
Label_4.place(x=330, y=199, width=80, height=30)
# 创建组件'Entry_3'及其设置
Entry_3 = tk.Entry(ts, text='', fg='black', bg='white', state='normal', relief='sunken', bd=1, font=('微软雅黑', 9), cursor='xterm')
Entry_3.place(x=410, y=200, width=38, height=29)
# 创建组件'Label_5'及其设置
Label_5 = tk.Label(ts, text='特征提取   ', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('方正小标宋简体', 9, ), cursor='arrow')
Label_5.place(x=88, y=264, width=80, height=30)
# 创建组件'Checkbutton_2'及其设置
Checkbutton_2 = tk.Checkbutton(ts, text='统计特征', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('微软雅黑', 9), cursor='arrow')
Checkbutton_2.place(x=189, y=264, width=110, height=30)
# 创建组件'Checkbutton_3'及其设置
Checkbutton_3 = tk.Checkbutton(ts, text='时域特征', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('微软雅黑', 9), cursor='arrow')
Checkbutton_3.place(x=338, y=264, width=110, height=30)
# 创建组件'Checkbutton_4'及其设置
Checkbutton_4 = tk.Checkbutton(ts, text='频域特征', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('微软雅黑', 9), cursor='arrow')
Checkbutton_4.place(x=491, y=264, width=110, height=30)
# 创建组件'Label_7'及其设置
Label_7 = tk.Label(ts, text='特征选择  ', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('方正小标宋简体', 9, ), cursor='arrow')
Label_7.place(x=88, y=323, width=80, height=30)
# 创建组件'Entry_4'及其设置
Entry_4 = tk.Entry(ts, text='', fg='black', bg='white', state='normal', relief='sunken', bd=1, font=('微软雅黑', 9), cursor='xterm')
Entry_4.place(x=285, y=328, width=37, height=25)
# 创建组件'Label_8'及其设置
Label_8 = tk.Label(ts, text='K值', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('微软雅黑', 9), cursor='arrow')
Label_8.place(x=189, y=323, width=80, height=30)
# 创建组件'Radiobutton_1'及其设置
Radiobutton_1 = tk.Radiobutton(ts, text='f_classif',variable=var,value='f_classif', fg='black', bg='#A8A8A8', state='normal', relief='sunken', bd=2, font=('微软雅黑', 9), cursor='arrow')
Radiobutton_1.place(x=381, y=323, width=110, height=30)
# 创建组件'Radiobutton_2'及其设置
Radiobutton_2 = tk.Radiobutton(ts, text='chi2',variable=var,value='chi2', fg='black', bg='#A8A8A8', state='normal', relief='sunken', bd=2, font=('微软雅黑', 9), cursor='arrow')
Radiobutton_2.place(x=381, y=353, width=110, height=30)
# 创建组件'Radiobutton_3'及其设置
Radiobutton_3 = tk.Radiobutton(ts, text='f_regression', fg='black',variable=var,value='f_regression', bg='#A8A8A8', state='normal', relief='sunken', bd=2, font=('微软雅黑', 9), cursor='arrow')
Radiobutton_3.place(x=554, y=323, width=110, height=30)
# 创建组件'Label_9'及其设置
Label_9 = tk.Label(ts, text='分类模型   ', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('方正小标宋简体', 9, ), cursor='arrow')
Label_9.place(x=88, y=403, width=80, height=30)
# 创建组件'Radiobutton_4'及其设置
Radiobutton_4 = tk.Radiobutton(ts, text='朴素贝叶斯',variable=var1,value='朴素贝叶斯',  fg='black', bg='#A8A8A8', state='normal', relief='sunken', bd=2, font=('微软雅黑', 9), cursor='hand2')
Radiobutton_4.place(x=161, y=403, width=110, height=30)
# 创建组件'Radiobutton_5'及其设置
Radiobutton_5 = tk.Radiobutton(ts, text='逻辑回归', variable=var1,value='逻辑回归', fg='black', bg='#A8A8A8', state='normal', relief='sunken', bd=2, font=('微软雅黑', 9), cursor='hand2')
Radiobutton_5.place(x=381, y=403, width=110, height=30)
# 创建组件'Radiobutton_6'及其设置
Radiobutton_6 = tk.Radiobutton(ts, text='随机森林', variable=var1,value='随机森林', fg='black', bg='#A8A8A8', state='normal', relief='sunken', bd=2, font=('微软雅黑', 9), cursor='hand2')
Radiobutton_6.place(x=271, y=403, width=110, height=30)
# 创建组件'Label_10'及其设置
Label_10 = tk.Label(ts, text='输出模型   ', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('方正小标宋简体', 9,), cursor='arrow')
Label_10.place(x=88, y=625, width=80, height=30)
# 创建组件'Label_11'及其设置
Label_11 = tk.Label(ts, text='输出数据   ',fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('方正小标宋简体', 9, ), cursor='arrow')
Label_11.place(x=88, y=556, width=80, height=30)
# 创建组件'Entry_5'及其设置
Entry_5 = tk.Entry(ts, text='',textvariable=entry_var4, fg='black', bg='white', state='normal', relief='sunken', bd=1, font=('微软雅黑', 9), cursor='xterm')
Entry_5.place(x=180, y=625, width=403, height=30)
# 创建组件'Entry_6'及其设置
Entry_6 = tk.Entry(ts, text='', textvariable=entry_var3,fg='black', bg='white', state='normal', relief='sunken', bd=1, font=('微软雅黑', 9), cursor='xterm')
Entry_6.place(x=177, y=556, width=406, height=30)
# 创建组件'Button_1'及其设置
Button_1 = tk.Button(ts, text='高级选项', fg='black', bg='white', state='normal', relief='raised', bd=2, font=('微软雅黑', 9), cursor='arrow')
Button_1.place(x=180, y=725, width=80, height=30)
# 创建组件'Button_2'及其设置
Button_2 = tk.Button(ts, text='运行', fg='black', bg='white',state='normal', relief='raised', bd=2, font=('微软雅黑', 9), cursor='arrow',command=lambda: start_task2() if var2.get()=="预测" else start_task())
Button_2.place(x=381, y=725, width=80, height=30)
# 创建组件'Label_6'及其设置
Label_6 = tk.Label(ts, text='当前模式', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('方正小标宋简体', 9, ), cursor='arrow')
Label_6.place(x=81, y=470, width=80, height=30)
# 创建组件'Radiobutton_7'及其设置
Radiobutton_7 = tk.Radiobutton(ts, text='训练', variable=var2,value='训练', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('微软雅黑', 9), cursor='hand2')
Radiobutton_7.place(x=150, y=470, width=90, height=30)
# 创建组件'Radiobutton_8'及其设置
Radiobutton_8 = tk.Radiobutton(ts, text='预测', variable=var2,value='预测', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('微软雅黑', 9), cursor='hand2')
Radiobutton_8.place(x=280, y=470, width=90, height=30)

# 创建组件'Label_12'及其设置
Label_12 = tk.Label(ts, text='评分函数', fg='black', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('微软雅黑', 9), cursor='arrow')
Label_12.place(x=301, y=323, width=80, height=30)

Label_13 = tk.Label(ts, text='地下工程  v2.0', fg='#00FF00', bg='#A8A8A8', state='normal', relief='flat', bd=2, font=('楷体', 30, 'italic underline'), cursor='arrow')
Label_13.place(x=698, y=695, width=436, height=77)
# 创建组件'Button_3'及其设置
Button_3 = tk.Button(ts, text='...', fg='black', bg='white', state='normal', relief='raised', bd=2, font=('微软雅黑', 9), cursor='arrow',command=lambda:select_file(1))
Button_3.place(x=618, y=78, width=31, height=30)
# 创建组件'Button_4'及其设置
Button_4 = tk.Button(ts, text='...', fg='black', bg='white', state='normal', relief='raised', bd=2, font=('微软雅黑', 9), cursor='arrow',command=lambda:select_file(2))
Button_4.place(x=618, y=135, width=31, height=30)
# 创建组件'Button_5'及其设置
Button_5 = tk.Button(ts, text='...', fg='black', bg='white', state='normal', relief='raised', bd=2, font=('微软雅黑', 9), cursor='arrow',command=lambda:select_file(3))
Button_5.place(x=618, y=556, width=31, height=30)
# 创建组件'Button_6'及其设置
Button_6 = tk.Button(ts, text='...', fg='black', bg='white', state='normal', relief='raised', bd=2, font=('微软雅黑', 9), cursor='arrow',command=lambda:select_file(4))
Button_6.place(x=618, y=625, width=31, height=30)


# 重定向 stdout 到 Text 控件
sys.stdout = StdoutRedirector(Message_1)

# 测试输出
print("欢迎使用地下工程-时序分析分类工具version2.0")


# 设置窗口循环（请勿删除）
ts.mainloop()


