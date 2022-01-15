# default_exp pixelmap
# > Pixelmaps have custom construction and will take more care of what i-cells are allowed to be contracted/removed.

from cProfile import label
from custom_dict_gmap import dict_nGmap
from pixel_map_z_curve_full import D, alpha_0, R, C, alpha_1, alpha_2
import matplotlib.pyplot as plt
from combinatorial.notebooks.combinatorial.zoo import G2_SQUARE_BOUNDED, G2_SQUARE_UNBOUNDED
from combinatorial.notebooks.combinatorial.gmaps import nGmap
import numpy as np
import logging
from random import random
import random
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# export


# -------------------------------- PixelMap class ----------------------------------------
class PixelMap (nGmap):
    """2-gMap representing an RxC image grid"""

    @property
    def n_rows(self):
        return self._nR

    @property
    def n_cols(self):
        return self._nC

    # vertex coordinates for the 8 darts in one pixel
    vertices = np.fromstring("""\
    0 0 1 0 2 0 2 1 2 2 1 2 0 2 0 1
    """, sep=' ', dtype=np.float32).reshape(8, 2)
    vertices -= 1
    vertices *= .95
    vertices += 1
    vertices /= 2
    vertices -= .5

    # text offsets relative to darts
    text_offsets = np.fromstring("""\
    0 0 -1 -1   0  0 1 1
    1 1  0  0  -1 -1 0 0
    """, sep=" ").reshape(2, 8).T

    text_angles = [0, 0, -90, -90, 0, 0, -90, -90]
    text_HAs = 'center center right right center center left left'.split()
    text_VAs = 'top top center center bottom bottom center center'.split()

    def _plot_dart_no(self, dart, rotate=False):
        if dart >= 8*self.n_rows*self.n_cols:
            return  # TODO plot the boundary darts, too, if the maps is unbounded
        vi0, vi1 = dart % 8, 1 + dart % 8
        if vi1 == 8:
            vi1 = 0
        verts = PixelMap.vertices[[vi0, vi1]]
        verts += [(dart // 8) % self.n_cols, (dart // 8) // self.n_cols]
        mid = .5 * verts[0] + .5 * verts[1]
        mid += 0.005 * PixelMap.text_offsets[vi0]
        plt.text(mid[0], mid[1], dart,
                 ha=PixelMap.text_HAs[dart % 8],
                 va=PixelMap.text_VAs[dart % 8],
                 rotation=PixelMap.text_angles[dart % 8] * rotate
                 )

    def plot_faces(self, number_darts=True):
        vertices = PixelMap.vertices
        # iterate over 2-cells
        for some_dart in self.darts_of_i_cells(2):
            x, y = [], []
            # for 2D maps the orbiting darts of the face are 'sorted'
            for dart in self.cell_2(some_dart):
                x.append(vertices[dart % 8, 0] + (dart // 8) % self.n_cols)
                y.append(vertices[dart % 8, 1] + (dart // 8) // self.n_cols)
                if number_darts:
                    self._plot_dart_no(dart)
            x.append(vertices[some_dart % 8, 0] +
                     (some_dart // 8) % self.n_cols)
            y.append(vertices[some_dart % 8, 1] +
                     (some_dart // 8) // self.n_cols)

            plt.fill(x, y, alpha=.2)
            plt.plot(x, y)
            plt.scatter(x[1::2], y[1::2], marker='+', color='k')
            plt.scatter(x[0::2], y[0::2], marker='o', color='k')

        plt.gca().set_aspect(1)
        plt.xticks([])  # (np.arange (self.n_cols))
        plt.yticks([])  # (np.arange (self.n_rows)[::-1])
        plt.ylim(self.n_rows-.5, -.5)
        plt.axis('off')
        plt.title(self.__str__())

    @classmethod
    def from_shape(cls, R, C, sew=True, bounded=True):
        """Constructs grid-like gmap from number rows and columns

        Args:
            R: number of rows
            C: number of columns
            sew: sew the pixels together (default) or not?
            bounded: set to False to add the outer boundary

        Returns:
            2-gMap representing a pixel array

        """
        def _swap2(A,  r1, c1, i1,  r2, c2, i2):
            """swap helper to 2-sew darts"""
            tmp = A[r1, c1, i1, 2].copy()
            A[r1, c1, i1, 2] = A[r2, c2, i2, 2]
            A[r2, c2, i2, 2] = tmp

        def _iter_boundary(R, C):
            """counter-clockwise boundary iteration around the block darts"""
            c = 0
            for r in range(R):
                yield 8*(r*C+c) + 7
                yield 8*(r*C+c) + 6
            r = R-1
            for c in range(C):
                yield 8*(r*C+c) + 5
                yield 8*(r*C+c) + 4
            c = C-1
            for r in range(R-1, -1, -1):
                yield 8*(r*C+c) + 3
                yield 8*(r*C+c) + 2
            r = 0
            for c in range(C-1, -1, -1):
                yield 8*(r*C+c) + 1
                yield 8*(r*C+c) + 0

        # set the members
        cls._nR = R
        cls._nC = C

        # compute the total number of darts
        n_all_darts = 8*R*C + (not bounded)*4*(R+C)

        # TODO dtype can be smaller for small images
        alphas_all = np.full((3, n_all_darts), fill_value=-1, dtype=np.int64)
        alphas_block = alphas_all[:, :8*R*C]  # view at the block part
        # view at the outer boundary part
        alphas_bound = alphas_all[:,  8*R*C:]

        # create the square by replicating bounded square with increments
        alphas_square = nGmap.from_string(G2_SQUARE_BOUNDED).T
        # rearrange view at the block part
        A = alphas_block.T.reshape((R, C, 8, 3))
        for r in range(R):
            for c in range(C):
                A[r, c] = alphas_square + 8 * (r*C + c)

        if sew:  # 2-sew the squares
            for r in range(R):
                for c in range(C-1):
                    _swap2(A, r, c, [2, 3], r, c+1, [7, 6])
            for c in range(C):
                for r in range(R-1):
                    _swap2(A, r, c, [4, 5], r+1, c, [1, 0])

        if not bounded:  # ` add boundary darts
            # set alpha0 to: 1 0 3 2 5 3 ...
            alphas_bound[0, 1::2] = np.arange(0, alphas_bound.shape[1], 2)
            alphas_bound[0, 0::2] = np.arange(1, alphas_bound.shape[1], 2)

            # set alpha1 to: L 2 1 4 3 ... 0
            alphas_bound[1, 0] = alphas_bound.shape[1]-1
            alphas_bound[1, 1:-1:2] = np.arange(2, alphas_bound.shape[1], 2)
            alphas_bound[1, 2:-1:2] = np.arange(1, alphas_bound.shape[1]-1, 2)
            alphas_bound[1, -1] = 0

            # add offsets to alpha0 and alpha1 of the boundary block
            alphas_bound[:2] += 8*R*C

            # 2-sew the the darts of the boundary with the darts of the block
            for d_bound, d_block in enumerate(_iter_boundary(R, C)):
                alphas_block[2, d_block] = d_bound + 8*R*C
                alphas_bound[2, d_bound] = d_block

        return cls.from_alpha_array(alphas_all)

    @classmethod
    def from_implicit_given_shape(cls, R, C, sew=True, bounded=False):
        """
            This method is useful to have the same set of darts generate by the implicit
            implementation using the Morton code. The basic implementation of the PixelMap
            give us a set of darts where they are sequential and do not follow the bit
            flip logic. Consequently, also the alphas will be wrong without this method.
        """
        m = dict_nGmap(2, D)

        return m


def is_self_adjacent(G, d):
    # returns if face is self touching at dart d
    #
    if G.a2(d) == d:
        return False  # border dart (needed only for bounded faces?)

    e = G.a2(d)  # 2-oposit of d

#   return set (G.cell_2 (d)) == set (G.cell_2 (e))   # compares two sets, which is not necessary
#   return                d   in set (G.cell_2 (e))   # TODO: can be done w/o creating the set
    return d in G.cell_2(e)    # only iteration ;)


def pendant_darts(G):
    for d in G.darts:
        if G.ai(1, d) == G.ai(2, d):
            yield d


# -------------------------------- LabelMap class ----------------------------------------
class LabelMap (PixelMap):
    # _initial_dart_polylines_00 stores start and end coordinates of darts in pixel (0,0)
    _initial_dart_polylines_00 = np.fromstring("""\
        0 1  2 1   2 2  2 2   2 1  0 1   0 0  0 0
        0 0  0 0   0 1  2 1   2 2  2 2   2 1  0 1
        """, sep=' ', dtype=np.float32).reshape(2, 16).T.reshape(8, 2, 2)
    _initial_dart_polylines_00 -= 1
    _initial_dart_polylines_00 *= .95
    _initial_dart_polylines_00 += 1
    _initial_dart_polylines_00 /= 2
    _initial_dart_polylines_00 -= .5

    @classmethod
    def from_labels(cls, labels):
        if type(labels) == str:
            n_lines = len(labels.splitlines())
            labels = np.fromstring(
                labels, sep=' ', dtype=np.uint8).reshape(n_lines, -1)
        # 'c' is the representation of the current Gmap
        c = cls.from_shape(labels.shape[0], labels.shape[1], bounded=False)

        '''
            Compunting the following command, I will not have anymore a LabelMap, but with my method
            I will have a dict_Gmap and I will lost all the properties of the LabelMap. Thus,
            my method is not the best option I have, I think...
            c = cls.from_implicit_given_shape(R,C,bounded=False) '''

        #print('c: ',c)
        cls._labels = labels
        #print('cls: ', cls)

        # add drawable polyline for each dart
        c._dart_polyline = {}
        for d in D:
            c._dart_polyline[d] = LabelMap._initial_dart_polylines_00[d % 8].copy()
            c._dart_polyline[d][..., 0] += (d // 8) % C
            c._dart_polyline[d][..., 1] += (d // 8) // C

        return c

    def plot(self, number_darts=True, image_palette='gray'):
        """Plots the label map.

        image_palette : None to not show the label pixels.
        """
        # self.darts is inehrited from the nGmap class in gmaps.py that generates an incremental sequence of darts.
        # For that reason, I want to swap self.darts with D, that is the set generated by the morton encoding.
        # for d in self.darts:
        for d in D:
            #e = self.a0(d)
            e = alpha_0(d)
            plt.plot(self._dart_polyline[d][:, 0],
                     self._dart_polyline[d][:, 1], 'k-')
            plt.plot([self._dart_polyline[d][-1, 0], self._dart_polyline[e][-1, 0]],
                     [self._dart_polyline[d][-1, 1], self._dart_polyline[e][-1, 1]], 'k-')
            # f = self.a1(d)
            # plt.plot ([self._dart_polyline[d][ 0,0],self._dart_polyline[f][ 0,0]],[self._dart_polyline[d][ 0,1],self._dart_polyline[f][ 0,1]], 'b-')
            if number_darts:
                self._plot_dart_no(d)
            plt.scatter(self._dart_polyline[d][0, 0],
                        self._dart_polyline[d][0, 1], c='k')
#             plt.scatter(self._dart_polyline[d][-1,0], self._dart_polyline[d][-1,1], marker='+')

        if image_palette:
            plt.imshow(self.labels, alpha=0.5, cmap=image_palette)

        plt.gca().set_aspect(1)
        plt.xticks([])  # (np.arange (self.n_cols))
        plt.yticks([])  # (np.arange (self.n_rows)[::-1])
        plt.ylim(self.n_rows-.5, -.5)
        plt.axis('off')
        plt.title(self.__str__())

    @property
    def labels(self):
        return self._labels

    def value(self, d):
        """Returns label value for given dart"""
        p = d // 8
        return self.labels[p // self.n_cols, p % self.n_cols]

    def remove_edges(self):
        # TODO edge removal causes skips in the outer loop if used w/i list() ???
        # d ... some dart while iterating over all edges
        for d in list(self.darts_of_i_cells(1)):
            # e ... dart of the oposit face
            e = self.a2(d)

            if d == e:                               # boundary edge
                logging.debug('Skipping: belongs to boundary.')
                continue
            if d == self.a1(e):                      # dangling dart
                logging.debug(f'{d} : pending')
#                 logging.info (d)
                self.remove_edge(d)
                continue
            if d == self.a0(self.a1(self.a0(e))):  # dangling edge
                logging.debug(f'{d} : pending')
#                 logging.info (d)
                self.remove_edge(d)
                continue
            if d in self.cell_2(e):                 # bridge (self-touching face)
                logging.debug(f'Skipping bridge at {d}')
                continue
            if (self.value(d) == self.value(e)).all():       # identical colour in CCL
                logging.debug(f'{d} : low-contrast')
#                 logging.info (d)
                self.remove_edge(d)
                continue
            logging.debug(f'Skipping: contrast edge at {d}')

    def remove_vertex(self, d):
        if not self.is_i_removable(0, d):
            return
        for d in self.cell_0(d):
            e = self.a0(d)
            self._dart_polyline[e] = np.vstack(
                (self._dart_polyline[e], self._dart_polyline[d][::-1]))
        super().remove_vertex(d)

    # TODO vetrices removal causes skips in darts in the outer loop if used w/i list() ???
    def remove_vertices(self):
        # d ... some dart while iterating over all vertices
        for d in list(self.darts_of_i_cells(0)):
            try:
                # the degree is checked inside
                self.remove_vertex(d)
                logging.debug(f'{d} removed')
            except:
                logging.debug(f'{d} NOT removable')


class custom_LM(dict_nGmap):

    def __init__(self):
        self.labels = {}
        self.m = dict_nGmap(2, D)
        self.boundary_darts = set()
        self.bounded = False

    def generate_chessboard_labels_odd_resolution(self):
        """
            I can distinguish three cases:
            cont = 0 -> bounding faces -> brown
            cont = odd values  -> B
            cont = even values -> W
        """

        cnt = 0
        for x in self.m.all_i_cells(2):

            print(list(x))
            for i in x:
                if cnt == 0:
                    self.labels[i] = 'brown'
                elif cnt % 2 == 0:
                    self.labels[i] = 'blue'
                else:
                    self.labels[i] = 'black'

            cnt += 1

    def generate_chessboard_labels_even_resolution(self, bounded=None):

        self.bounded = bounded
        res = R*C
        if res % 2 != 0:
            self.generate_chessboard_labels()
        else:
            k = 1
            cnt = 0
            odd_label = 'gray'
            even_label = 'black'
            for x in self.m.all_i_cells(2):

                print(list(x))
                # 1 + (R*k_odd) is the point from which a new row starts
                if cnt == 1+((R/2)*k) or cnt == 2+((R/2)*k):
                    if k % 2 == 0:
                        odd_label = 'gray'
                        even_label = 'black'
                    else:
                        odd_label = 'black'
                        even_label = 'gray'

                    k += 1
                    # print(cnt)

                for i in x:
                    if cnt == 0:
                        self.labels[i] = 'brown'
                    elif cnt % 2 == 0:
                        self.labels[i] = even_label
                    else:
                        self.labels[i] = odd_label
                cnt += 1

        if bounded == True:

            for d, l in zip(self.labels.keys(), self.labels.values()):
                if l == 'brown':
                    self.boundary_darts.add(d)
                    # self.m.darts.remove(d)

            for bd in self.boundary_darts:
                self.labels[bd] = self.labels[alpha_2(bd)]

    def generate_chessboard_labels_considering_alpha2(self, bounded=None):

        #self.bounded = bounded
        boundary = False  # not done yet
        label = 'black'

        for x in self.m.all_i_cells(2):
            for i in x:
                if boundary == False:
                    self.labels[i] = 'brown'
                    continue

                try:
                    if self.labels[alpha_2(i)] == 'black':
                        label = 'blue'
                    elif self.labels[alpha_2(i)] == 'blue':
                        label = 'black'
                except KeyError:
                    pass

            for i in x:
                if boundary == False:
                    self.labels[i] = 'brown'
                    continue
                self.labels[i] = label

            boundary = True  # just to avoid the if above during next iterations

    def generate_chessboard_random_labels(self):

        labels = ['green', 'red', 'blue', 'violet', 'orange']

        for x in self.m.all_i_cells(1):
            c = random.choice(labels)
            for i in x:
                self.labels[i] = c

    def _i_remove_contract(self, i, dart, rc, skip_check=False):
        """
        Remove / contract an i-cell of dart
        d  ... dart
        i  ... i-cell
        rc ... +1 => remove, -1 => contract
        skip_check ... set to True if you are sure you can remove / contract the i-cell
        """
        logging.debug(
            f'{"Remove" if rc == 1 else "Contract"} {i}-Cell of dart {dart}')

        if not skip_check:
            assert self._is_i_removable_or_contractible(i, dart, rc),\
                f'{i}-cell of dart {dart} is not {"removable" if rc == 1 else "contractible"}!'

        """
            Every time the checking is over I assume that I can increment the variable that keeps trace of
            the levels because the dart is removed/contracted, for sure.
        """
        self.m.level += 1
        #print(f'level -> {self.level}')

        # print(type(self.alpha[i]))

        i_cell = set(self.m.cell_i(i, dart))  # mark all the darts in ci(d)
        print(i_cell)

        #print(f'i-cell -> {i_cell}')
        #print(f'\n{i}-cell to be removed {i_cell}')
        for d in i_cell:
            if self.labels[dart] != self.labels[d]:
                print(
                    f'{dart} and {d} do not have the same label! I cannot set the new alpha for the dart {d}')
                continue

            d1 = self.m.ai(i, d)  # d1 ← d.Alphas[i];
            if d1 not in i_cell:  # if not isMarkedNself(d1,ma) then
                # d2 ← d.Alphas[i + 1].Alphas[i];
                #print(f'd1 -> {d1}')
                d2 = self.m.ai(i+rc, d)
                d2 = self.m.ai(i, d2)
                while d2 in i_cell:  # while isMarkedNself(d2,ma) do
                    # d2 ← d.Alphas[i + 1].Alphas[i];
                    d2 = self.m.ai(i+rc, d2)
                    d2 = self.m.ai(i, d2)
                logging.debug(
                    f'Modifying alpha_{i} of dart {d1} from {self.m.ai (i,d1)} to {d2}')

                self.m.set_ai(i, d1, d2)  # d1.Alphas[i] ← d2;

        """
            In that for loop, in addition to remove the dart given in input,
            I also check if involutions of this dart are different from the
            implicitly ones. If they are different, then they will move into
            the passive part, otherwise nothing will be stored.

            I wanto to say that implementation is been provided to work with
            2-Gmaps. For that reason there are only three if statements to check
            which j-value I am analysing.
        """
        for d in i_cell:  # foreach dart d' ∈ ci(d) do

            if self.labels[dart] != self.labels[d]:
                print(
                    f'{dart} and {d} do not have the same label! I cannot remove dart {d}')
                continue

            for j in self.m.all_dimensions:
                if j == 0:
                    if self.m.alpha[j][d] != self.m.alpha[j].get_alpha0(d):
                        self.m.custom_alpha[j][d] = self.m.alpha[j][d]

                if j == 1:
                    if self.m.alpha[j][d] != self.m.alpha[j].get_alpha1(d):
                        self.m.custom_alpha[j][d] = self.m.alpha[j][d]

                if j == 2:
                    if self.m.alpha[j][d] != self.m.alpha[j].get_alpha2(d):
                        self.m.custom_alpha[j][d] = self.m.alpha[j][d]

            self.m.dart_level[d] = self.m.level

            #print(f'dart that will be removed -> {d}')
            self.m._remove_dart(d)  # remove d' from gm.Darts;

    def _is_i_removable_or_contractible(self, i, dart, rc):
        """
        Test if an i-cell of dart is removable/contractible:

        i    ... i-cell
        dart ... dart
        rc   ... +1 => removable test, -1 => contractible test
        """
        assert dart in self.m.darts
        assert 0 <= i <= self.m.n
        assert rc in {-1, +1}

        if rc == +1:  # removable test
            if i == self.m.n:
                return False
            if i == self.m.n-1:
                return True
        if rc == -1:  # contractible test
            if i == 0:
                return False
            if i == 1:
                return True

        for d in self.m.cell_i(i, dart):
            if self.m.alpha[i+rc][self.m.alpha[i+rc+rc][d]] != self.m.alpha[i+rc+rc][self.m.alpha[i+rc][d]]:
                return False
        return True

    def remove_edges(self):

        # d ... some dart while iterating over all edges
        for d in list(self.m.darts_of_i_cells(1)):

            #print(f'd: {d}',end='-')
            # e ... dart of the oposit face
            try:
                e = self.m.alpha[2][d]
            except KeyError:
                e = alpha_2(d)

            print(f'd:{d}, e:{e}')

            """
                In the case in which the generated chessboard is not bounded (bounded = False), the boundary darts set will be empty
                and this if statment will be skipped.
            """
            if e in self.boundary_darts and self.bounded == False:
                continue

            """
                Condition to check if d is a boundary dart. I have assigned BROWN label to the boundary of the chessboard just to represent
                a real one.
            """
            if self.labels[d] == 'brown':                               # boundary edge
                logging.debug('Skipping: belongs to boundary.')
                print(f'{d} is a boundary dart')
                continue

            """
                If the label of d and e are the same, the edge at which the dart d belong, can be removed.
            """
            if self.labels[d] == self.labels[e] and self.labels[d] == self.labels[alpha_0(d)] and self.labels[d] == self.labels[alpha_0(alpha_2(d))]:
                print(
                    f'I am removing the dart {d} due to the same colorful label!')
                self.m._remove(1, d)
                continue

            if d == alpha_1(e):
                print(f'{d} is a pending dart')
                self.m._remove(1, d)
                continue

    def remove_edges_from_label(self, label):

        for d in list(self.m.darts_of_i_cells(1)):

            """
                    If the label of d and e are the same, the edge at which the dart d belong, can be removed.
                """
            if self.labels[d] == label and self.labels[d] == self.labels[alpha_0(d)] and self.labels[d] == self.labels[alpha_0(alpha_2(d))]:
                print(
                    f'I am removing the dart {d} due to the same colorful label!')
                self.m._remove(1, d)
                continue

    def generated_grafted_grids(self):
        
        cnt = 0
        labels = ['brown', 'green', 'orange', 'orange', 'green']
        k = 1
        l = labels[k]
        #toggle_val = 1

        ''' for _ in range(4):
            toggle_val = (1,0)[toggle_val]
            print(toggle_val) '''


        for x in self.m.all_i_cells(2):
            for i in x:
                if cnt == 0:
                    self.labels[i] = 'brown'
                    continue
                
                

                if cnt == 1+(R*k):
                    print(cnt)
                    k += 1
                    try:
                        l = labels[k]
                    except IndexError:
                        pass
                self.labels[i] = l
                    
            cnt += 1

