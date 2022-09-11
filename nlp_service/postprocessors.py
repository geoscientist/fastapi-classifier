import pandas as pd 
import numpy as np


def no_oper_processor(df):
    no_oper_idx = df[df['total_prediction'].str.contains('no_oper')].index
    bna_keywords = ['прием', 'внесен', 'внес', 'приём', 'внести', 'приним', 'взнос', 'депозит']
    ch_keywords = ['выда', 'снят', 'выдает', 'выдаёт', 'сним']
    pay_keywords = ['платеж', 'оплат']

    no_oper_BNA_idx = df[df['total_prediction'].str.contains('no_oper') & 
                       df['Комментарий'].apply(lambda x: True if any(word in x for word in bna_keywords) else False) &
                      (df['Комментарий'].apply(lambda x: True if any(word in x for word in ch_keywords) else False) == False)].index

    no_oper_CH_idx = df[df['total_prediction'].str.contains('no_oper') & 
                       df['Комментарий'].apply(lambda x: True if any(word in x for word in ch_keywords) else False) &
                      (df['Комментарий'].apply(lambda x: True if any(word in x for word in bna_keywords) else False) == False)].index

    no_oper_combined_idx = df[df['total_prediction'].str.contains('no_oper') & 
                      df['Комментарий'].apply(lambda x: True if any(word in x for word in bna_keywords) else False) & 
                      df['Комментарий'].apply(lambda x: True if any(word in x for word in ch_keywords) else False)].index

    no_oper_pay_idx = df[df['total_prediction'].str.contains('no_oper') & 
                       df['Комментарий'].apply(lambda x: True if any(word in x for word in pay_keywords) else False)].index

    no_oper_idx = [idx for idx in no_oper_idx if idx not in (list(no_oper_BNA_idx) +
                                          list(no_oper_CH_idx) +
                                          list(no_oper_combined_idx) +
                                          list(no_oper_pay_idx))]


    df['sub_target'] = df['total_prediction'].apply(lambda x: ','.join([pred for pred in x.split(',') if pred not in ['no_oper']]))
    df.at[no_oper_idx,'sub_target'] = df.loc[no_oper_idx,'sub_target'] + ',no_oper'
    df.at[no_oper_BNA_idx,'sub_target'] = df.loc[no_oper_BNA_idx,'sub_target'] + ',no_oper_BNA'
    df.at[no_oper_CH_idx,'sub_target'] = df.loc[no_oper_CH_idx,'sub_target'] + ',no_oper_CH'
    df.at[no_oper_combined_idx,'sub_target'] = df.loc[no_oper_combined_idx,'sub_target'] + ',no_oper'
    df.at[no_oper_pay_idx,'sub_target'] = df.loc[no_oper_pay_idx,'sub_target'] + ',no_oper_payment'
    
    df['sub_target'] = df.sub_target.apply(lambda x: ','.join([pred for pred in x.split(',') if pred != '']))
    return df

def cassette_processor(df):
    # this processor has disadvantage - it overwrites another classes
    df.loc[df['Комментарий'].str.contains('CASSETTE TYPE') | df['Комментарий'].str.contains('cassette type'), 'total_prediction'] = 'cassette'
    df.loc[df['Комментарий'].str.contains('CASSETTE TYPE') | df['Комментарий'].str.contains('cassette type'), 'sub_target'] = 'cassette'
    return df