import random
count = 0
for k in range(10000):
    arr = [1, 1, 1]
    minus = random.randint(0, 2)
    arr[minus] = 0
    arr_new = arr[0::]
    first_choose = random.randint(0, 2)
    for m in range(0, 3):
        if m != minus and arr[m] == 1:
            arr.pop(m)
            break
    arr.pop(first_choose)
    new = arr[0]
    if new == 0:
        count+=1
