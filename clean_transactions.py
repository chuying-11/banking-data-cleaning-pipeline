import pandas as pd
import logging
import sys
import argparse
import os

biz_key=['客戶ID', '交易日期_clean', '交易金額_clean', '交易類型_clean', '分行代碼']

os.makedirs('logs', exist_ok=True)
os.makedirs('output', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='清洗交易資料並驗證輸出') #description:做說明，只在跑--help時出現
    parser.add_argument('--input', default='data/transactions.csv', help='輸入檔路徑') #default:使用者沒提供這個選項時的替補值
    parser.add_argument('--output', default='output/transactions_clean.csv', help='輸出檔路徑')
    return parser.parse_args()

logging.basicConfig(
    level=logging.INFO,  #低於此等級的忽略
    format='%(asctime)s [%(levelname)s] %(message)s', # %(asctime)s→事件發生時間、%(levelname)s→此訊等級名、 %(message)s→呼叫時回傳的話
    handlers=[  #訊息的出口清單
        logging.FileHandler('logs/pipeline.log', encoding='utf-8'),  #寫進檔案
        logging.StreamHandler()  #印到螢幕(終端機)
    ]
)

# --------------------------
#讀入檔案
def load_data(path):
    df=pd.read_csv(path, dtype={'客戶ID':str, '分行代碼':str}, na_values=['-', ' '])
    logging.info(f"讀入資料：{len(df)}筆")
    return df


#日期處理step1~4
def clean_dates(df):
    df['交易日期_clean']=pd.NaT  #1 建立空欄

    mask_cn=df['交易日期'].str.contains('年', na=False)  #2 中文處理mask
    df.loc[mask_cn, '交易日期_clean']=pd.to_datetime(df.loc[mask_cn, '交易日期'], format='%Y年%m月%d日')  #寫進'交易日期_clean'
    
    mask_slash=df['交易日期'].str.contains('/', na=False)  #3 斜線處理mask
    first=df['交易日期'].str.split('/').str[0]   #取每筆第一段
    mask_roc=mask_slash & (first.str.len()==3)  #民國mask
    mask_west=mask_slash & (first.str.len()==4) #西元mask
    roc_fixed=(  #把民國變西元
        (first[mask_roc].astype(int)+1911).astype(str)
        + '/'
        + df.loc[mask_roc, '交易日期'].str.split('/', n=1).str[1]  #n=1：只切一次
    )
    df.loc[mask_roc, '交易日期_clean']=pd.to_datetime(roc_fixed, format='%Y/%m/%d')  #寫進'交易日期_clean'
    df.loc[mask_west, '交易日期_clean']=pd.to_datetime(df.loc[mask_west, '交易日期'], format='%Y/%m/%d')

    mask_hyphen=df['交易日期'].str.contains('-', na=False)  #4 橫線處理mask
    iso=df['交易日期'].str.split('-').str[0]  #ISO mask
    mask_iso=mask_hyphen & (iso.str.len()==4)
    eu=df['交易日期'].str.split('-').str[2]  #EU mask
    mask_eu=mask_hyphen & (eu.str.len()==4)  #無橫線者str[2]為 NaN → len比較回傳False → 再由mask_hyphen攔截
    df.loc[mask_iso, '交易日期_clean']=pd.to_datetime(df.loc[mask_iso, '交易日期'], format='%Y-%m-%d') 
    df.loc[mask_eu, '交易日期_clean']=pd.to_datetime(df.loc[mask_eu, '交易日期'], format='%d-%m-%Y') #寫進'交易日期_clean'
    logging.info(f"日期清洗完成：{df['交易日期_clean'].notna().sum()}筆有效，{df['交易日期_clean'].isna().sum()}筆空值")
    return df


#金額處理step1~5
def clean_amounts(df):
    df['交易金額_clean']=df['交易金額'].str.normalize('NFKC')  #1 全形轉半形
    df['交易金額_clean']=df['交易金額_clean'].str.strip() #2 去空格
    df['交易金額_clean']=df['交易金額_clean'].str.replace(',', '', regex=False) #3 去逗號
    df['交易金額_clean']=df['交易金額_clean'].str.replace('NT$', '', regex=False) #4 去NT$
    df['交易金額_clean']=df['交易金額_clean'].astype(float) #5 astype轉數值
    logging.info(f"金額清理完成：總額{df['交易金額_clean'].sum()}")
    return df


#交易類型處理
def clean_types(df):
    df['交易類型_clean']=df['交易類型'].str.strip() #去空格
    logging.info(f"類型種類數：{df['交易類型_clean'].nunique()}種")
    return df


#清除重複列
def deduplicate(df):
    df_final = df.drop_duplicates()
    logging.info(f"去重完成：刪除{len(df) - len(df_final)}筆，剩{len(df_final)}筆")
    return df_final


#資料驗證
def validate(df):
    #1 列數：
    logging.info(f"本批列數：{len(df)}")
    assert len(df) > 0, '空檔案：讀入0筆'
    
    #2 日期空值：交易日期_clean的 NaT數
    logging.info(f"日期空值:{df['交易日期_clean'].isna().sum()} 筆")
    assert df['交易日期_clean'].notna().sum() > 0, '日期全數解析失敗'
    
    #3 日期範圍：min >= 1990-01-01 且 max <= 今日(<明日零點)
    today = pd.Timestamp.today().normalize()  #.today().normalize()：今日零點
    assert df['交易日期_clean'].max() < (today + pd.Timedelta(days=1)), f"日期異常：出現未來日期{df['交易日期_clean'].max()}"
    assert df['交易日期_clean'].min() >= pd.Timestamp('1990-01-01'), f"日期異常：出現疑似未轉換的古代日期 {df['交易日期_clean'].min()}"
    
    #4 金額對帳：兩欄總額差 < 0.01
    assert abs(df['交易金額'].str.replace(r'[^0-9０-９]', '', regex=True).astype(float).sum() - df['交易金額_clean'].sum()) < 0.01, '金額對帳異常'
    
    #5 金額下限：min > 0
    assert df['交易金額_clean'].min()>0, '金額下限異常'
    
    #6 類型白名單：  # <= (⊆) 用在兩個集合之間意為「是否為子集」
    assert set(df['交易類型_clean'].dropna()) <= {'存款', '提款', '轉帳', '消費'}, f"白名單異常：異常類型{set(df['交易類型_clean'].dropna()) - {'存款', '提款', '轉帳', '消費'}}"
    
    #7 識別碼格式：客戶ID全7碼純數字、分行代碼全3碼純數字
    assert (df['客戶ID'].str.fullmatch(r'\d{7}')).all(), '客戶ID格式異常'
    assert (df['分行代碼'].str.fullmatch(r'\d{3}')).all(), '分行代碼格式異常'
    
    #8 完全重複：df.duplicated().sum() == 0
    assert df.duplicated().sum() == 0, f"完全重複異常：預期0，實際{df.duplicated().sum()}"
    
    #9 業務重複監控：已知待查，報數即可
    logging.warning(f"業務重複待查：{df.duplicated(subset=biz_key).sum() - df.duplicated().sum()}筆")

    logging.info('全部檢查通過 ✓')


#呼叫全部
def main():
    args = parse_args()
    try:
        df = load_data(args.input)
        df = clean_dates(df)
        df = clean_amounts(df)
        df = clean_types(df)
        df_final = deduplicate(df)
        validate(df_final)
        df_final.to_csv(args.output, index=False, encoding='utf-8-sig')
        logging.info(f"已輸出:{args.output}，共{len(df_final)}筆")
    except FileNotFoundError as e:
        logging.error(f"找不到輸入檔：{e}")
        sys.exit(1)
    except AssertionError as e:
        logging.error(f"資料驗證失敗：{e}")
        sys.exit(1)


if __name__ == '__main__': 
    main()



