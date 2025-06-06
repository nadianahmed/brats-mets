def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    '''
    Call in a loop to create terminal progress bar

    Parameters:
    - iteration(Int): current iteration
    - total(Int): total iterations
    - prefix(String): prefix string
    - suffix(String): suffix string
    - decimals(Int): positive number of decimals in percent complete
    - length(Int): character length of bar
    - fill(String): bar fill character
    - printEnd(String): end character (e.g. "\r", "\r\n")
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)

    if iteration == total: 
        print(f'{prefix} |{bar}| {100.0}% {suffix}', end = " ✅\n")