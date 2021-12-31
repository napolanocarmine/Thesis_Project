from pixel_map_z_curve_full import alpha_1, alpha_0, alpha_2

class implicit_dict(dict):
    
    def __init__(self, i = None):
        super().__init__()
        self.i = i

    def __getitem__(self, k):
        
        #print(f'k -> {k}')
        # i have commented this if because it generates some KeyErrors and I have thought of
        # merging it with the other below because, in my opinion, the behaviour should be the same
        """ if k not in self.flag_set:
            print(f'k -> {k}')
            raise KeyError """

        if k not in super().keys() :#or k not in self.flag_set:
            #print(f'sono in get item -> {self.i}')
            if self.i == 0:
                return self.get_alpha0(k) #vedere come gestire per un generico alpha
            elif self.i == 1:
                return alpha_1(k)
            else:
                return self.get_alpha2(k)

        return super().__getitem__(k)

    def __setitem__(self, k, v):
        
        #self.flag_set.add(k)

        if self.i == 0:
            if self.get_alpha0(k) != v:
                #print('alpha 0 diverso')
                return super().__setitem__(k, v)

        elif self.i  == 1:
            if alpha_1(k) != v:
                #print('alpha 1 diverso')
                return super().__setitem__(k, v)
        
        elif self.i == 2:
            if self.get_alpha2(k) != v:
                #print('alpha 2 diverso')
                return super().__setitem__(k, v)


    def __delitem__(self, v):
        
        #print(self.flag_set)

        # the following operation could be useless because when i arrive here, i have already removed the v dart
        #self.flag_set.remove(v)

        if v in super().keys():
            print('sto cancellando...')
            return super().__delitem__(v)

    def get_alpha0(self, k):
        return alpha_0(k)
    
    def get_alpha1(self, k):
        return alpha_1(k)

    def get_alpha2(self, k):
        return alpha_2(k)

    def set_i(self, i):
        self.i = i
