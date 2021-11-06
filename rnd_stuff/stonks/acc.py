def total_return(p2, p1, portfolio, fee):
    ret = (p2+p2*fee)/p1 - 1
    return sum(ret*portfolio)
