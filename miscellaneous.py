from scipy.stats import truncnorm

def truncnorm_standard(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low-mean) / sd,
                     (upp-mean) / sd,
                     loc=mean,
                     scale=sd)
