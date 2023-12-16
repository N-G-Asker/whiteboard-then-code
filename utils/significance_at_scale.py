
def get_Wilson_CI(p_hat: float, n: int=650, confidence_level:int=95):
    """
    Calculates the Wilson confidence interval (CI).

    The idea to use the Wilson CI comes from the paper
    "SELF-REFINE: Iterative Refinement with Self-Feedback"
    (https://arxiv.org/pdf/2303.17651.pdf), where they use this computation to determine 
    statistical significance when comparing performance across models on the same
    metric/benchmark. See "Appendix J: Statistical Confidence Intervals" in the paper 
    for more details. (Note: Confusingly, the results they report in the table appear 
    to assume the Wilson Confidence Interval is centered about the raw success rate p_hat 
    -- that is, the estimate value obtained directly from evaluation, e.g., solve rate -- 
    when in fact it is centered about an adjusted value of p_hat.)

    The code for the computation is based on 
    https://stackoverflow.com/questions/10029588/
    "Python implementation of the Wilson Score Interval?" and was validated 
    against a NIST resource
    (https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) and the 
    original paper cited in the SELF-REFINE paper that recommends the method:
        Brown, L. D. Cai, T. T. and DasGupta, A. (2001). Interval estimation for a 
        binomial proportion", Statistical Science, 16(2), 101-133.  

    Args:
        p_hat: 
            The fraction of passed questions over the total attempted (`n`), as a float.
            In particular, the pass@5 rate is used here. Alternative names: 
            "solve rate", "pass rate".

        n: 
            The total number of questions attempted as an int. Defaults to 650.

        confidence_level: An integer in {85, 90, 95, 99}. Defaults to 95%

    """
    confidence_levels = {
        85: 1.44,
        90: 1.645,
        95: 1.96,
        99: 2.576
    }
    
    if confidence_level not in confidence_levels.keys():
        print("Argument value error: Confidence level must be an integer in {85, 90, 95, 99}.")
        raise ValueError

    if n == 0:
        return 0
    
    z = confidence_levels[confidence_level] # e.g., sets the z-score to be 1.96 for a 95% CI

    lower = ( (p_hat + z**2 / (2 * n) - z * ((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)**0.5) /
                (1 + (z**2 / n)) )

    upper = ( (p_hat + z**2 / (2 * n) + z * ((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)**0.5) /
                (1 + (z**2 / n)) )

    interval = (lower, upper)
    midpoint = (upper - lower) / 2 + lower
    offset = upper - midpoint
    return midpoint, offset, interval


# res = get_Wilson_CI(args.p_hat, args.n, args.c)
# print(res)

rows = []
rows.append("16.84 & 21.18 & 23.63 & 25.35 & 26.73".split(' & '))
rows.append("11.06 & 13.95 & 15.62 & 16.80 & 17.67".split(' & '))
rows.append("11.03 & 14.55 & 16.47 & 17.88 & 19.05".split(' & '))
rows.append("10.03 & 13.69 & 15.78 & 17.20 & 18.31".split(' & '))
rows.append("8.28 & 11.02 & 12.66 & 13.84 & 14.77".split(' & '))
rows.append("8.49 & 11.35 & 13.08 & 14.34 & 15.38".split(' & '))
rows.append("8.95 & 12.14 & 14.26 & 15.94 & 17.38".split(' & '))
rows.append("8.12 & 11.14 & 13.02 & 14.31 & 15.23".split(' & '))
rows.append("8.58 & 11.77 & 13.71 & 15.11 & 16.15".split(' & '))

round = 0
for row in rows:
    round += 1
    print(round)
    line = ""
    for num in row:
        res = get_Wilson_CI(float(num)/100)
        line += f"{res[0]*100:.2f} $\pm$ {res[1]*100:.2f} & "
    print(line)
