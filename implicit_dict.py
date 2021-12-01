from pixel_map_z_curve_full import alpha_1, D

class implicit_dict(dict):
    
    def __init__(self):
        super().__init__()
        self.flag_set = D

    def __getitem__(self, k):

        if k not in self.flag_set:
            raise KeyError

        if k not in super().__getitem__(k):
            return alpha_1(k) #vedere come gestire per un generico alpha

        return super().__getitem__(k)

    def __setitem__(self, k, v):

        self.flag_set.add(k)

        #the following if has to be written in order to can manage the different alphas
        if alpha_1(k) != v:
            return super().__setitem__(k, v)

    def __delitem__(self, v):

        self.flag_set.remove(v)

        if v in super().__getitem__(v):
            return super().__delitem__(v)

    def get_set(self):
        return self.flag_set

