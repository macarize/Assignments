valid_input = \
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]

invalid_input = \
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]

def check_global(sudoku):
    for i in range(len(sudoku)):
        for j in range(len(sudoku[0])):
            if sudoku[i][j] > -1:
                loc = (i, j)
                value = sudoku[i][j]
                for k in range(len(sudoku[0])):
                    if value == sudoku[i][k] and loc != (i, k):
                        print("row-wise duplicate {} ".format(value) + "in {}".format((i, k)))
                    sudoku_T = [[sudoku[j][i] for j in range(len(sudoku))] for i in range(len(sudoku[0]))]
                    if value == sudoku_T[j][k] and loc != (k, j):
                        print("column-wise duplicate {} ".format(value) + "in {}".format((k, j)))
def partition(sudoku, r, c):
    partitioned = []
    partitioned_sudoku = []
    for i in range(r, r+3):
        temp = []
        for j in range(c, c+3):
            temp.append(sudoku[i][j])
        partitioned_sudoku.append(temp)
    partitioned.append(partitioned_sudoku)
    return partitioned

def convert(sudoku):
    for i in range(len(sudoku)):
        for j in range(len(sudoku[0])):
            if sudoku[i][j] == ".":
                sudoku[i][j] = -1
            sudoku[i][j] = int(sudoku[i][j])
    return sudoku

def check_local(sudoku):
    seen = []
    for i in range(len(sudoku)):
        for j in range(len(sudoku)):
            if sudoku[i][j] is not -1:
                seen.append(sudoku[i][j])
    if len(seen) != len(set(seen)):
        print("duplicates in 3x3 box")
    else:
        print("No duplicates")



invalid_input = convert(valid_input)
check_global(invalid_input)
k = (0, 3, 6)

partitioned = []
for j in k:
    for i in k:
        partitioned_sudoku = partition(invalid_input, j, i)
        partitioned.append(partitioned_sudoku[0])
for i in range(0,9):
    print("checking {}".format(i) + "th 3x3 box")
    check_local(partitioned[i])
