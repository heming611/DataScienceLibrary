def t_test(sample_treatment, sample_control):
    """
    sample_treatment: a list or an array for the treatment sample
    sample_control: a list or an array ofr the control sample
    """
    n_t, n_c = len(sample_treatment), len(sample_control)
    mu_hat_t, mu_hat_c = np.mean(sample_treatment), np.mean(sample_control)
    sigma2_hat_t = 1.0 / (n_t - 1) * np.sum((sample_treatment - mu_hat_t) ** 2)
    sigma2_hat_c = 1.0 / (n_c - 1) * np.sum((sample_control - mu_hat_c) ** 2)

    t = (mu_hat_t - mu_hat_c) * 1.0 / np.sqrt(sigma2_hat_t / n_t + sigma2_hat_c / n_c)

    return round(t, 4)


def p_value(t, alternative_hypothesis="t!=c"):
    """
    alternative_hypothesis: a string indicating the alternative hypothesis, take values 't<c', 't>c', 't!=c'
    """
    if alternative_hypothesis == "t!=c":
        p = 2 * round(1 - norm.cdf(abs(t)), 4)
    elif alternative_hypothesis == "t<c":
        p = round(norm.cdf(t), 4)
    else:
        p = round(1 - norm.cdf(t), 4)

    return ps