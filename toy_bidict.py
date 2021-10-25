from typing import Tuple
from bidict import bidict

############################## toy example of the bidict library ##############################
print('############################## toy example of the bidict library ##############################')
element = bidict({'H' : 'hydrogen'})
print(element)
print(element['H'])
print(element.inverse['hydrogen'])



############################## trying to store a list as value of the bidict ##############################


#element_list = [1,2,3]
#element2 = bidict({'H' : element_list})
#print(element2)

#anagrams_by_alphagram = bidict(opt=['opt', 'pot', 'top'])
#print(anagrams_by_alphagram)

"the execution give me this error -> unhashable type: 'list'"


############################## example using a tuple ##############################
print('\n############################## example using a tuple ##############################')
tupla = (('opt','opt2','opt3'), 'pot', 'top')
anagrams_by_alphagram = bidict(opt=tuple)
print(anagrams_by_alphagram)
print(anagrams_by_alphagram['opt'])
print(anagrams_by_alphagram.inverse[tuple])

"if a want to update the tuple, i have to redifine the bidict"
new_element = 'opt1'
tupla = tupla + (new_element,)
anagrams_by_alphagram.update(opt=tuple)
print(anagrams_by_alphagram.inverse[tuple])
#print(type(anagrams_by_alphagram['opt'][0]))

print(tupla[0][2])


############################## test to have the same value for two different keys ##############################
""" test_bidict = bidict(a=1, b=1)
print(test_bidict)
print(test_bidict['a'])
print(test_bidict['b'])
print(test_bidict.inverse['1']) """



############################## test to have the same key for two different values ##############################
""" print('############################## test to have the same key for two different values ##############################')
tupla = (1,2)
test_bidict = bidict(a=tupla, b=tupla)
print(test_bidict)
print(test_bidict['a'])
print(test_bidict.inverse[tupla]) """



############################## structure of the one possibile kind of implementation for Gmap ##############################
"""
Structure of the bidict for Gmap

    dart_i: key of the data structure
    dart_j, level, involution: tuple that contains the related dart, at which level of pyramid, type of the involution 
"""

print('############################## structure of the one possibile kind of implementation for Gmap ##############################')

content = ('dart_j', 'level', 'alpha_i')
gmap_bidict = bidict(dart_i = content)
gmap_bidict.put('dart1', 'cose2')

print(gmap_bidict)
print(gmap_bidict['dart_i'])
print(gmap_bidict.inverse[content])



print('##### test #####')
n = 3
dart = None 
alpha = tuple(None for _ in range(n+1))
print(alpha)
pair = (dart, alpha)
bidict_gmap = bidict({ dart : pair})

print(bidict_gmap)



############################## initialization of nGmap ##############################

n = 3
alpha_set = [bidict() for _ in range(n + 1)]
print(f'alpha initialization: {alpha_set}')