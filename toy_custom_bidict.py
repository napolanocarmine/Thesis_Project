class MyBidict(dict):
    def __init__(self, init_dict):
            self.inverse = {}
            dict.__init__(self, init_dict)
            for key, value in init_dict.items():
                self.inverse[value] = key

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.inverse.__setitem__(value, key)


job_bidict = MyBidict({"John": "director", "Mike": "designer",
                      "Anna": "designer", "Lisa": "engineer"})
print(job_bidict["Mike"])
# output => designer
print(job_bidict.inverse["designer"])
# output => Anna


class MyBidict2:
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
        return self.dict_normal[key]

    def get_item_inverse(self, value):
        return self.dict_inverse[value] 

job_bidict = MyBidict2({"John":"director", "Mike":"designer", "Anna":"designer", "Lisa":"engineer"})
print(job_bidict.get_item("Mike"))
# output => ['designer']
print(job_bidict.get_item_inverse("designer"))
# output => ['Mike', 'Anna']
