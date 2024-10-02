def mean(num_list):
    return sum(num_list) / len(num_list)


def add_five(num_list):
    return [n+5 for n in num_list]

print("testing mean function")
n_list = [34,44,23,46,12,24]
print("from inside useful functions "  + str(mean(n_list)))