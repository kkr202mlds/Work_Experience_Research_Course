def get_hand_value(*args):
    sum, n_elv = 0, 0
    for i in args:
        sum += i
        if i == 11:
            n_elv += 1
    if sum <= 21:
        for i in args:
            if i == 11 or i == 10:
                return 'Winner Winner Chicken Dinner!'
            else:
                return 'The value of your hand is {}'.format(sum)
    elif sum > 21  and n_elv > 0:
        t = sum
        for i in range(n_elv):
            t -= 10
        if t <= 21:
            return 'The value of your hand is {}'.format(t)
        else:
            return 'Busted!'
    elif sum > 21 and n_elv == 0:
        return 'Busted!'

# While testing the execution of your program, I will use the hands below:
print(get_hand_value(4,4,4,4))
print(get_hand_value(11,10))
print(get_hand_value(11,5,6,4,11,4))
print(get_hand_value(11,5,6,10))
print(get_hand_value(10,5,7))
print(get_hand_value(5,6,5,10))
print(get_hand_value(10,11))
print(get_hand_value(11,11,11,11,10,10))
print(get_hand_value(5,6,5,5))
print(get_hand_value(21,1))

"""
Test Cases
Failed: 1, Passed: 0 of 1 tests
test_exercise
Your code failed this test
Error details
'Winner Winner Chicken Dinner!' != 'The value of your hand is 21'
- Winner Winner Chicken Dinner!
+ The value of your hand is 21
 : Hand 2: Output should be 'Winner Winner Chicken Dinner!'"""
"""'The value of your hand is 21' != 'Busted!'
- The value of your hand is 21
+ Busted!
 : Hand 3: Output should be 'The value of your hand is 21'
"""
"""(Challenge exercise) Function and logic practice
Casino 21: You are given a random number of integers with values ranging from 1 to 11. These are passed in to your function as ints. In python you can accept an uncertain number of arguments in functions like this using the *args parameter in the function. The function treats these parameter objects as a tuple. So if you passed in 4,5,6,7 as arguments when calling your function, the function would receive a tuple (4,5,6,7) as its parameter. You are required to write a function which takes the value of these integers as *args and works according to the following specifications:

a) If the sum of all the integers are less than or equal to 21, return their sum in the form of a string using the .format() method to embed the value of the sum. If this returned value is printed, and the value were 20 for example, the output would look like this The value of your hand is 20.

b) If their sum exceeds 21 and there are one or more 11's present among the integers, reduce the total sum by 10. If the sum is still over 21, check to see if there are more 11's and if so, reduce the sum by 10 again. 

c) If the sum exceeds 21 and no reductions are possible (either because there were no 11's to begin with or 10 has been reduced from the value of each 11 that was present already), return 'Busted!'.

d) If the value of the integers add up to 21 and there is one integer 11 and another integer 10, return 'Winner Winner Chicken Dinner!'"""
