def quicksort(nums, start, end):
    if start >= end:
        return
    key = nums[end]
    i, j = start, end
    while i<j:
        while nums[i]<key and i<j:
            i+=1
        nums[i], nums[j] = nums[j], nums[i]
        while nums[j]>=key and i<j:
            j-=1
        nums[j], nums[i] = nums[i], nums[j]
    nums[i] = key
    quicksort(nums, start, i-1)
    quicksort(nums, i+1, end)
arr = [4,5,6,3,2,4,5]
quicksort(arr, 0, 6)
print(arr)