from combinatorial.notebooks.combinatorial.custom_gmaps import nGmap
from bidict import bidict

class MyBidict():
    def __init__(self, init_dict = None):
        self.dict_normal = {}  # normal dict to store (key, values)
        self.dict_inverse = {}  # normal dict to store (value. keys)
        if init_dict != None:
            for key in init_dict.keys():
                self.add_item(key, init_dict[key])

    def add_item(self, key, value):
        if key in self.dict_normal:
            self.dict_normal[key].append(value)
        else:
            self.dict_normal[key] = [value]
        if value in self.dict_inverse:
            self.dict_inverse[value].append(key)
        else:
            self.dict_inverse[value] = [key]

    def get_item(self, key):
        """Get value by key

        Args:
            key ([int]): key you want to see the associated values

        Returns:
            [dict]: dict of the value associated to the inserted key
        """
        #print(self.dict_normal)
        return self.dict_normal[key]

    def get_item_inverse(self, value):
        return self.dict_inverse[value] 

    def get_normal_dict(self):
        return self.dict_normal

    def get_inverse_dict(self):
        return self.dict_inverse

    def delete_item(self, key):
        self.dict_normal.pop(key)





class my_Gmaps(nGmap, MyBidict):
    def __init__(self, array):
        super().__init__(array)
        # self.n = n
        # self.alpha = [bidict() for _ in range(n + 1)]
        self._n = 3
        self._alpha = [MyBidict() for _ in range(self._n + 1)]
        

    #override method
    def set_ai(self, i, dart, new_dart):
        """Sets dart.alpha_i = new_dart"""
        """ assert 0 <= i <= self.n
        self[i, dart] = new_dart
 """
        assert 0 <= i <= self.n
        if new_dart != -1:
            print(dart)
            print(new_dart)
            self._alpha[i].add_item(dart, new_dart)
        #print(f'There is not a new dart for dart {dart}.\n')

    def test_method(self):
        print('pippo')

    def save_to_bidict(self, d, new_d):
        """Save the information about removal operation into the bidict

        Args:
            d ([dart]): is the dart you have choose to remove
            new_d ([dart]): is the new dart to link
        """

        print('sono qui')
        # scegliere il tipo di bidict alpha da usare per salvare

    def get_dict(self, i):
        return self._alpha[i].get_normal_dict()

    def get_all_dict(self):
        for i in range(self._n):
            print(self._alpha[i].get_normal_dict())
            print(self._alpha[i].dict_inverse)

