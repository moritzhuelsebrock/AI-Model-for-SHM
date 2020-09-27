import numpy as np
import sys
# List
classmate = ['sam','bob','john']
lady = ['adele','jean']

# Add element
classmate.append('jane')
classmate.insert(1,'ana')
classmate.extend(lady)
# print(classmate)

# Delete element
classmate.pop(0)
classmate.remove('ana')
# print(classmate)

# Dictionary
d1 = {'sam' : 1.0, 'jack': 2.7, 'jason':5.0}
d1["sam"]=1.3
List1=d1.keys()
List2=d1.values()
# print(List1)

# Add/update element
d1.update({'xu':2.0,'sam':1.7})
# print(d1,"length:",len(d1))

# Delete element
d1.pop('xu')
d1.clear()
# print(d1)

# Loop Statement
# for x in range(2,9,3):
#     print(x)

# Function
def power(x,n=0):
    y=1
    for i in range(n):
        y=y*x
    print('power x=',y)

# Function with Arbitrary Argument *Args
def calc(*numbers):
    sum=0
    for i in numbers:
        sum=sum+i*i
    print("calc x=",sum)

# Function with Arbitrary Keyword Argument **kwargs
def person(name,age,**kw):
    print('name:',name,'age:',age,'other:',kw)
# person('sam',20,dota='100')

def student(name,age,*,city='Beijing',dota):
    print("name:",name,"age:",age,"city:",city,"dota:",dota)
# student('xu',25,dota='ancient')

# Recursion
def fact(x):
    if x==1:
        return 1
    return x*fact(x-1)

def fact_opt(x,product):
    if x==1:
        return product
    else:
        return fact_opt(x-1,x*product)

# List Comprehension
x = [x*x for x in range(10)]
d={'sam':'sb',"xu":'nb',"laoxu":'wd'}
l=["samxu",'world',18,'apple',None]
# print([s.upper() if isinstance(s,str) else s for s in l])

# Generator
y=(y*y for y in range(10))

def fib(max):
    n,a,b=0,0,1
    while n<max:
        yield b
        a,b=b,a+b
        n=n+1

def odd():
    print('Step 1')
    yield 1
    print('Step 2')
    yield 3
    print('Step 3')
    yield 5

def triangles(max):
    n=0
    l=[1]
    while n<max:
        yield l
        l = [0]+l+[0]
        l = [l[i]+l[i+1] for i in range(len(l)-1)]
        n=n+1

if __name__=='__main__':
    o=triangles(3)
    for i in o:
        print(i)

# Object Oriented Programming
class Student(object):
    def __init__(self,name,score):
        self.name=name
        self.score=score
    def print_score(self):
        print('%s:%s'%(self.name,self.score))
Lisa = Student('Lisa',18)
Lisa.dota='Ancient'
# Lisa.print_score()

# F String Formating
w={"sam":"ancient","john":4000}
print(f'MMR of {w["sam"]} is {w["john"]}')