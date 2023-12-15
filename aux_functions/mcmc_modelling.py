import pymc as pm
import arviz as az

def get_specified_dfs(df_complete, variables_of_dfs_to_build):
    dfs = []
    for name, variables in variables_of_dfs_to_build.items():
        dfs.append(df_complete[variables].copy())
    return dfs

def get_models(dfs, models_names, y, y_proposal_distribution="normal"):
    models = []
    for df, model_name in zip(dfs, models_names):
        with pm.Model(coords={'beta_names': df.columns.values}, name=model_name) as model:
            # Priors
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10, shape=df.shape[1], dims="beta_names") # Number of betas varies with number of covariates in the dataset
            sigma = pm.HalfNormal("sigma", sigma=10)
            
            # Expected value of outcome
            mu = alpha + pm.math.dot(df, beta)
            
            if _is_proposal_distribution_cauchy(y_proposal_distribution):
                gamma = pm.HalfNormal("gamma", sigma=10) # For Cauchy
                Y_obs = pm.Cauchy("Y_obs", alpha=mu, beta=gamma, observed=y) # In the docs, the gamma is called beta
            
            if _is_proposal_distribution_student(y_proposal_distribution): 
                nu = pm.HalfNormal("nu", sigma=10) # For TStudent
                Y_obs = pm.StudentT("Y_obs", mu=mu, sigma=sigma, nu=nu, observed=y)
            
            if _is_proposal_distribution_normal(y_proposal_distribution): 
                Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y)
        models.append(model)
    return models

def _is_proposal_distribution_cauchy(y_proposal_distribution):
    return True if y_proposal_distribution == "cauchy" else False

def _is_proposal_distribution_student(y_proposal_distribution):
    return True if y_proposal_distribution == "student_t" else False

def _is_proposal_distribution_normal(y_proposal_distribution):
    return True if y_proposal_distribution == "normal" else False

def sample_models(num_samples, models):
    sampling_results = []
    for i, model in enumerate(models):
        with model:
            step = pm.Metropolis()
            sample = _sample_model(num_samples, step)
            sampling_results.append(sample)
    return sampling_results

def _sample_model(num_samples, step):
    return pm.sample(num_samples, step=step, idata_kwargs = {'log_likelihood': True})

# This creates a dictionary with each model name and each sampling result
def get_identified_sampling_results(sampling_results, models_names):
    model_trace_dict = dict() # {'model_name': sample_of_model}
    for i, num in enumerate(models_names):
        model_trace_dict.update({num: sampling_results[i]})
    return model_trace_dict

def get_waic_measures(identified_sampling_results):
    return az.compare(identified_sampling_results, ic='waic')

def get_model_from_list(models, model_name):
    for model in models:
        if model.name == model_name:
            return model