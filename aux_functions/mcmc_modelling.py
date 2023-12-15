import pandas as pd
import arviz as az
import pymc as pm

def get_list_with_specified_dfs(df_complete):
    """
    Receives the complete dataset of the study
    Returns a list with 6 datasets with the variables specified in the report
    in the order: [complete, top_8, top_5, top_4, top_3, top_2]
    
    See variables of each in the report
    """
    df_top8 = df_complete.drop(columns=['age', 'region_1', 'region_2', 'region_3', 'region_4',]).astype(float)
    df_top5 = df_top8.drop(columns=['med_school_affil_1', 'med_school_affil_2', 'avelbl_services', 'ln_num_beds']).astype(float)
    df_top4 = df_top5.drop(columns='routine_xray_ratio').astype(float)
    df_top3 = df_top4.drop(columns='ln_avg_census').astype(float)
    df_top2 = df_top3.drop(columns='lenght_of_stay').astype(float)
    dfs = [df_complete, df_top8, df_top5, df_top4, df_top3, df_top2]
    return dfs

def get_list_with_pm_models(dfs):
    """Returns list of 6 models, for training"""
    models = []
    for df in dfs:
        models.append(pm.Model(coords={'beta_names': df.columns.values}))

    return models