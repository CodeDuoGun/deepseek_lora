from ssl import SO_TYPE
from typing import List


"""
给你一个整数数组 nums ，
判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。
请你返回所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。

输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
"""
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        特判，对于数组长度 n，如果数组为 null 或者数组长度小于 3，返回 []。
            对数组进行排序。
            遍历排序后数组：
            若 nums[i]>0：因为已经排序好，所以后面不可能有三个数加和等于 0，直接返回结果。
            对于重复元素：跳过，避免出现重复解
            令左指针 L=i+1，右指针 R=n−1，当 L<R 时，执行循环：
            当 nums[i]+nums[L]+nums[R]==0，执行循环，判断左界和右界是否和下一位置重复，去除重复解。并同时将 L,R 移到下一位置，寻找新的解
            若和大于 0，说明 nums[R] 太大，R 左移
            若和小于 0，说明 nums[L] 太小，L 右移

        """
        n=len(nums)
        res=[]
        if(not nums or n<3):
            return []
        nums.sort()
        res=[]
        for i in range(n):
            if(nums[i]>0):
                return res
            #去重，当起始的值等于前一个元素，那么得到的结果将会和前一次相同
            if(i>0 and nums[i]==nums[i-1]):
                continue
            L=i+1
            R=n-1
            while(L<R):
                # 如果等于0，将结果对应的索引位置的值加入结果集中
                if(nums[i]+nums[L]+nums[R]==0):
                    # 将三数的结果集加入到结果集中
                    res.append([nums[i],nums[L],nums[R]])
                    # 在将左指针和右指针移动的时候，先对左右指针的值，进行判断
                    # 如果重复，直接跳过
                    # 去重，因为 i 不变，当此时 l取的数的值与前一个数相同，所以不用在计算，直接跳
                    while(L<R and nums[L]==nums[L+1]):
                        L=L+1
                    # 去重，因为 i不变，当此时 r 取的数的值与前一个相同，所以不用在计算
                    while(L<R and nums[R]==nums[R-1]):
                        R=R-1
                    L=L+1
                    R=R-1
                elif(nums[i]+nums[L]+nums[R]>0): #如果结果大于0，将右指针左移
                    R=R-1
                else:#如果结果小于0，将左指针右移
                    L=L+1
        return res


ss = Solution()
print(ss.threeSum([-1,0,1,2,-1,-4]))