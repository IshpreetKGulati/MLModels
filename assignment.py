# class Solution:
#     def isValid(self, s: str) -> bool:
#         stack = []
#         for i in range(0, len(s)):
#             if len(s) == 0:
#                 return True
#             if s[i] == '[' or s[i] == '(' or s[i] == '{':
#                 stack.append(s[i])
#             if (s[i] == ']' or s[i] == ')' or s[i] == '}') and len(stack) == 0:
#                 return False
#             elif (s[i] == ']' and stack[len(stack) - 1] == '[') or (s[i] == '}' and stack[len(stack) - 1] == '{') or \
#                     (s[i] == ')' and stack[len(stack) - 1] == '('):
#                 stack.pop()
#
#             elif (s[i] == ']' and stack[len(stack) - 1] != '[') or (s[i] == '}' and stack[len(stack) - 1] != '{') or (
#                     s[i] == ')' and stack[len(stack) - 1] != '('):
#                 return False
#
#             if len(stack) == 0 and i == len(s) - 1:
#                 return True
# s = Solution()
# print(s.isValid('([)'))


"""class MyQueue:

    def __init__(self):

        self.stack1 = []
        self.stack2 = []

    def push(self, x: int) -> None:
       \
        self.stack1.append(x)
        return self.stack1

    def pop(self) -> int:

        if self.empty() is not True:
            while len(self.stack1) != 1:
                x = self.stack1.pop()
                self.stack2.append(x)

            x = self.stack1.pop()
            #print(len(self.stack1))
            self.stack1 = self.stack2[::-1]
            # print("pop", self.stack1)
            # print("pop", self.stack2)
            self.stack2 = []
        else:
            return
        return self.stack1

    def peek(self) -> int:

        if self.empty() is not True:
            while len(self.stack1) != 0:
                x = self.stack1.pop()
                self.stack2.append(x)
            # print("peek", self.stack2)
            # print("peek", self.stack1)
            self.stack1 = self.stack2[::-1]
            #print("peek", self.stack1)
            self.stack2 = []
            return x
        else:
            return

    def empty(self) -> bool:

        if len(self.stack1) == 0:
            return True
        else:
            return False

# Your MyQueue object will be instantiated and called as such:
obj = MyQueue()
x = [1,2,]
for i in range(0, len(x)):
    print(obj.push(x[i]))

param_3 = obj.peek()
print(param_3)
param_2 = obj.pop()
print(param_2)
param_2 = obj.pop()
print(param_2)
# param_2 = obj.pop()
# print(param_2)
# param_2 = obj.pop()
# print(param_2)
# param_2 = obj.pop()
# print(param_2)
# param_2 = obj.pop()
# print(param_2)

param_4 = obj.empty()
print(param_4)
"""


class Solution:

    def searchMatrix(self, matrix, target):

        left_outer = 0
        res = False
        right_outer = len(matrix)-1
        while left_outer < right_outer:

            mid_outer = (left_outer + right_outer ) // 2
            print("mid_outer ", mid_outer )
            N = len(matrix[0])-1

            if matrix[mid_outer][0] <= target and matrix[mid_outer][N] >= target:

                left_inner = 0
                right_inner = len(matrix[0])
                print("i", 1)
                while (left_inner < right_inner):
                    print(1)
                    mid_inner = left_inner + int((right_inner - left_inner) / 2)
                    print(mid_inner)
                    print(matrix[mid_outer][mid_inner])
                    if matrix[mid_outer][mid_inner] == target:
                        return True

                    if matrix[mid_outer][mid_inner] > target:
                        right_inner = mid_inner

                    elif matrix[mid_outer][mid_inner] < target:
                        left_inner = mid_inner + 1

            if matrix[mid_outer][0] < target:
                left_outer = mid_outer + 1

            else:
                right_outer = mid_outer

        return res

s = Solution()
matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]


target = 5
print((s.searchMatrix(matrix, target)))

class Solution:

    def searchMatrix(self, matrix, target):
        res = False
        rownumber = 0
        colnumber = len(matrix)-1
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return  res
        while rownumber < len(matrix) and colnumber >= 0:
            if matrix[rownumber][colnumber] == target:
                return True
            if target > matrix[rownumber][colnumber]:
                rownumber += 1

            elif target < matrix[rownumber][colnumber]:
                colnumber -= 1
        return res
s = Solution()
matrix = [[1,4]]

target = 4
print((s.searchMatrix(matrix, target)))


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        else:
            left_height = self.maxDepth(root.left)
            right_height = self.maxDepth(root.right)

        if left_height > right_height:
            return left_height + 1
        else:
            return right_height + 1




# import bisect
# def b(a,x,lo=0, hi=None):
#     print("f", 1)
#     print(a)
#     while lo < hi:
#         mid = (lo+hi)//2
#         if a[mid] < x: lo = mid+1
#         else: hi = mid
#     return lo
#
# class Solution(object):
#     def searchMatrix(self, matrix, target):
#         """
#         :type matrix: List[List[int]]
#         :type target: int
#         :rtype: bool
#         """
#         if not matrix or not matrix[0]: return False
#         for row in matrix:
#             if target <= row[-1] and target>=row[0]:
#                 i = b(row, target, 0, len(row))
#                 print(i)
#                 if i != len(row) and row[i] == target:
#                     return True
#         return False
#
# s = Solution()
# matrix =[[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
#
# target = 5
# print((s.searchMatrix(matrix, target)))
#
#
# class Solution:
#
#     def searchMatrix(self, matrix, target):
#
#         res = False
#         if len(matrix) == 0 or len(matrix[0]) == 0:
#             return False
#         for row in matrix:
#
#             left_inner = 0
#             right_inner = len(matrix[0]) - 1
#             while (left_inner <= right_inner):
#
#                 mid_inner = (left_inner + right_inner) // 2
#
#                 if row[mid_inner] == target:
#                     return True
#
#                 elif row[mid_inner] > target:
#                     right_inner = mid_inner - 1
#
#                 elif row[mid_inner] < target:
#                     left_inner = mid_inner + 1
#
#         return False
#
#
# s = Solution()
# matrix = [[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]]
#
# target = 5
# print((s.searchMatrix(matrix, target)))

# import heapq
#
#
# class Solution:
#     def findKthLargest(self, nums, k) :
#         # a = heapq.heapify(nums)
#         # print("A",a)
#         # nLarge = heapq.nlargest(k, nums)
#         # return (nLarge[k - 1])
#         nums.sort(reverse= True)
#         return nums[k-1]
# s = Solution()
# a = [3,6,8,1,95,5,2,4]
# print(s.findKthLargest(a,3))


class Node:
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def height(node):
    if node is None:
        return 0
    else:
        lheight = height(node.left)
        rheight = height(node.right)

    if lheight >rheight:
        return lheight+1
    else:
        return rheight+1

def fillMissingNodes(root, level):
    if (root == None):
        return
    if(root.left == None and level > 0):
        root.left = Node(-1)
    if(root.right == None and level > 0):
        root.right = Node(-1)
    fillMissingNodes(root.left, level - 1)
    fillMissingNodes(root.right, level - 1)

def BFS(root):

    b = []
    h = height(root)
    fillMissingNodes(root, h)
    for i in range(1, h+1):
      b = (printGivenLevel(root, i))

    return b, h

def printGivenLevel(root, level,a = []):

    if root is None:
        return
    if level == 1 or root.val == None:
        a.append(root.val)
    elif level > 1:
        printGivenLevel(root.left, level-1,a)
        printGivenLevel(root.right, level-1,a)
    return a
class Solution:
    def widthOfBinaryTree(self, root: Node) -> int:
        a, h = BFS(root)
        print(a)
        print(h)
        n = 2**h
        s = 2**(h-1)
        print(s,n)
        x = len(a)
        while True:
            if a[x-1] == -1:
                a.pop()
                x -= 1
            else:break
        return len(a[s-1:n:])


root = Node(1)

root.left = Node(3)
root.right = Node(2)

root.left.left = Node(5)
#root.left.right = Node(3)
# root.left.right = Node(None)
#root.right.left = Node(None)
#root.right.right = Node(9)

#root.left.left.left = Node(6)
# root.left.left.right = Node(None)
# root.left.right.right = Node(None)
# root.left.right.left = Node(None)
# root.right.left.left = Node(None)
# root.right.left.right = Node(None)
# root.right.right.left = Node(None)
#root.right.right.right = Node(9)

S = Solution()
print(S.widthOfBinaryTree(root))



