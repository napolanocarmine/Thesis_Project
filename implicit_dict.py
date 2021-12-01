from pixel_map_z_curve_full import alpha_1, D, alpha_0, alpha_2

class implicit_dict(dict):
    
    def __init__(self):
        super().__init__()
        self.flag_set = D
        #print(f'\nD\n{self.flag_set}')

    def __getitem__(self, k):

        if k not in self.flag_set:
            raise KeyError

        if k not in super().keys():
            return alpha_1(k) #vedere come gestire per un generico alpha

        return super().__getitem__(k)

    def __setitem__(self, k, v):

        self.flag_set.add(k)

        #the following if has to be written in order to can manage the different alphas
        if alpha_1(k) != v:
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

