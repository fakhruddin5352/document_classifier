import numpy as np

arr = [ 
        [[1,2,3],[4,5,6], [7,8,9]],
        [[10,11,12],[13,14,15],[16,17,18]]
      ]
arr1 = np.array(arr)
print(arr1)
arr1  =arr1/1.
sum = arr1[:,:,0]+arr1[:,:,1]+arr1[:,:,2]

print(sum)
arr1[:,:,0] = arr1[:,:,0]/sum
arr1[:,:,1] = arr1[:,:,1]/sum
arr1[:,:,2] = arr1[:,:,2]/sum

print(arr1)
