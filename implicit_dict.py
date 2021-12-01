from pixel_map_z_curve_full import D, alpha_1, alpha_0, alpha_2

class implicit_dict(dict):
    
    def __init__(self):
        super().__init__()
        self.flag_set = D
        #print(f'\nD\n{self.flag_set}')
        self.i = None

    def __getitem__(self, k):
        
        #print(f'k -> {k}')
        if k not in self.flag_set:
            print(f'k -> {k}')
            raise KeyError

        elif k not in super().keys():
            #print(f'sono in get item -> {self.i}')
            if self.i == 0:
                return self.get_alpha0(k) #vedere come gestire per un generico alpha
            elif self.i == 1:
                return alpha_1(k)
            else:
                return self.get_alpha2(k)

        return super().__getitem__(k)

    def __setitem__(self, k, v):
        
        self.flag_set.add(k)
        #print('sono entrato in set item')

        if self.i == 0:
            if self.get_alpha0(k) != v:
                return super().__setitem__(k, v)

        elif self.i  == 1:
            if alpha_1(k) != v:
                return super().__setitem__(k, v)
        
        elif self.i == 2:
            if self.get_alpha2(k) != v:
                return super().__setitem__(k, v)


    def __delitem__(self, v):
        
        #print(self.flag_set)

        # the following operation could be useless because when i arrive here, i have already removed the v dart
        #self.flag_set.remove(v)

        if v in super().keys():
            return super().__delitem__(v)

    
    def get_set(self):
        return self.flag_set

    def get_alpha0(self, k):
        return alpha_0(k)

    def get_alpha2(self, k):
        return alpha_2(k)

    def set_i(self, i):
        self.i = i
