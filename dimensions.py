output_size = [0,0,0,0]


def conv(ls_filter,ls_strides,padding):
    global output_size
    output_size[0] = output_size[0]
    output_size[1] = (output_size[1] + 2*padding - ls_filter[1])/ls_strides[1] + 1
    output_size[2] = output_size[1]
    output_size[3] = ls_filter[3]

def pool(ls_window,ls_strides,padding):
    global output_size
    output_size[1] = (output_size[1] + 2*padding - ls_window[1])/ls_strides[1] + 1
    output_size[2] = output_size[1]


def main():
    ls_pool = [1,30,20,10,5]
    ls_filters = [[3,3],[3,3],[3,3],[3,3]]
    for i in range(4):
        ls_filters[i].append(ls_pool[i])
        ls_filters[i].append(ls_pool[i+1])

    global output_size
    output_size = [100,28,28,1]
    for i in range(4):
        conv(ls_filters[i],[1,1,1,1],0)
        pool([1,2,2,1],[1,1,1,1],0)
        print(output_size)


main()