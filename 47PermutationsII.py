class Solution:
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        # if len(heights) == 1:
        #     return heights[0]
        # minx = heights.index(min(heights))
        # if minx == 0:
        #     count_min = max(heights[minx] * len(heights), self.largestRectangleArea(heights[minx+1::]))
        # elif minx == len(heights)-1:
        #     count_min = max(heights[minx] * len(heights), self.largestRectangleArea(heights[0:minx]))
        # else:
        #     count_min = max(heights[minx] * len(heights), self.largestRectangleArea(heights[0:minx]), self.largestRectangleArea(heights[minx+1::]))
        # return count_min

        ret = 0
        heights.append(0)
        index = []
        for i in range(len(heights)):
            while index.__len__()>0 and heights[index[-1]] >= heights[i]:
                h = heights[index[-1]]
                index = index[:-1]
                sidx = index[-1] if len(index)>0 else -1
                if h * (i-sidx-1)>ret:
                    ret = h * (i-sidx-1)
            index.append(i)
        return ret
s = Solution()
print(s.largestRectangleArea([5, 6, 4, 2, 3, 7]))


# def quicksort(nums, start, end):
#     if start >= end:
#         return
#     key = nums[end]
#     i, j = start, end
#     while i<j:
#         while nums[i]<key and i<j:
#             i+=1
#         nums[i], nums[j] = nums[j], nums[i]
#         while nums[j]>=key and i<j:
#             j-=1
#         nums[j], nums[i] = nums[i], nums[j]
#     nums[i] = key
#     quicksort(nums, start, i-1)
#     quicksort(nums, i+1, end)

# s = [4,5,6,3,2,4,5]
# quicksort(s, 0, 6)
# print(s)