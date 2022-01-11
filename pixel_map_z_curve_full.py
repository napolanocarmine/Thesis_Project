# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
# The following routines are for 32-bits
# TODO: have onese for 8, 16, and 64 bits
def part1by1(n):
    n&= 0x0000ffff
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n

def unpart1by1(n):
    n&= 0x55555555
    n = (n ^ (n >> 1)) & 0x33333333
    n = (n ^ (n >> 2)) & 0x0f0f0f0f
    n = (n ^ (n >> 4)) & 0x00ff00ff
    n = (n ^ (n >> 8)) & 0x0000ffff
    return n

def interleave2(x, y):
    return part1by1(x) | (part1by1(y) << 1)

def deinterleave2(n):
    return unpart1by1(n), unpart1by1(n >> 1)

# %% [markdown]
# # Border testing
# 
# To test if a dart is matching a specific pattern $P$ (binary representation),
# the following procedure can be applied
# 
# 1. construct xor number x
# 2. construct and number a
# 3. return not (d xor x) and a
# 
# The procedure to create the mask x
# 
# - `x`: replace `*` by `0`, e.g `*1*0*` --> `01000`
# - `a`: replace:
#     `*` -> `0`, otherwise `1` 
# 
# ## What are the gains?
# 
# - arbitrary number of rows and columns R x C
# - branch-less, i.e, no `if-then`, no `for` loops
# 

# %% [markdown]
# ## How many bits will the darts need?
# 
# - 3 bits (`bbb`) to encode the position of a dart within edge
# - bits to encode x,y in the interleaving fashion
# 
#     - we need to encode rows $r \in \{0..R\}$, and columns $c \in \{0..C\}$, i.e., including R and C respectively.
#     - the encoding can have two forms
# 
# | encoding   | condition |
# | ---------: | :-- |
# | xyxyx bbb   | (C+1) columns need more bits then the (R+1) rows |
# |  yxyx bbb   | otherwise |
# 
# ## Maximum cols/rows that can be encoded ...
# 
# ... by B-bit unsigned integers
# 
# Reserving the 3 rightmost bits to encode the 8 Nort-West darts of every pixel we have $B-3$ bits to encode the pixel coordinates. These are split as follows:
# 
# - $ B/2 - 2 $ bits for rows, and $0 < R < 2^{B/2 - 2}$
# - $ B/2 - 1 $ bits for columns, and $0 < C < 2^{B/2 - 1}$
# 
# Examples
# 
# 
# |  bits |  row bits | column bits | max Rows | max Columns | 
# | --: | --: | --: | --- | --- |
# |  8 |  2 |   3 |           3 |          7 | 
# | 16 |  6 |   7 |          63 |        127 | 
# | 32 | 14 |  15 |       16383 |      32767 | 
# | 64 | 30 |  31 |  1073741823 | 2147483647 | 

# %%
"""
for B in 8,16,32,64:
    print (f'| {B:2} | {B//2-2:2} |  {B//2-1:2} |  {2**(B//2-2)-1:10} | {2**(B//2-1)-1:10} | ')
"""
# %%
R,C = 4,4 # number of rows and columns

# %%
bits2type = {
    64 : np.uint64,
    32 : np.uint32,
    16 : np.uint16,
     8 : np.uint8,
}

# %%
def number_of_bits (n):
    'how many bits needed to represent a positive integer'
    return int (1 + np.floor (np.log2(n)))

# %%
nBitsX = number_of_bits(C)
nBitsY = number_of_bits(R)

if nBitsX  > nBitsY:
    nBits = 2 * nBitsX + 2  #     x y x y x b b b
else:
    nBits = 2 * nBitsY + 3  #   y x y x y x b b b


TYPE_UINT = None
for b,typ in bits2type.items():
    if nBits <= b: TYPE_UINT=typ

nBits, TYPE_UINT

# %%
# Masks
MASK_H, MASK_V = interleave2(2**nBitsX-1,0), interleave2(0,2**nBitsY-1)
MASK_C = MASK_H | MASK_V
MASK_I = 0

# %%
"""
print (f'''\
{MASK_C:0{nBits-3}b}  ... AND mask for corners
{MASK_V:0{nBits-3}b}  ... AND mask for top and bottom borders
{MASK_H:0{nBits-3}b}  ... AND mask for left and right borders
{MASK_I:0{nBits-3}b}  ... AND mask for interior darts
''')
"""

# %%
bitpatterns = np.array ([
# mask    x    y    3-bit     dx  dy       pattern example  comment
    
  MASK_C, 0,   0,   0b000,     0,  0,    #  e.g., 00000000, corner upper left  : horizontal dart 
  MASK_C, 0,   0,   0b100,     0,  0,    #  e.g., 00000100, corner upper left  : vertical   dart 
  MASK_C, C-1, 0,   0b001,     1,  0,    #  e.g., 00101001, corner upper right : horizontal dart 
  MASK_C, C,   0,   0b110,    -1,  0,    #  e.g., 10000110, corner upper right : vertical   dart 
  MASK_C, 0,   R,   0b010,     0, -1,    #  e.g., 01000010, corner lower left  : horizontal dart 
  MASK_C, 0,   R-1, 0b101,     0, +1,    #  e.g., 00010101, corner lower left  : vertical   dart 
  MASK_C, C-1, R,   0b011,    +1, -1,    #  e.g., 01101011, corner lower right : horizontal dart 
  MASK_C, C,   R-1, 0b111,    -1, +1,    #  e.g., 10010111, corner lower right : vertical   dart 
    
  MASK_V, C,   0,   0b000,    -1,  0,    #  e.g., *0*0*000, darts in border top, left
  MASK_V, C,   0,   0b001,    +1,  0,    #  e.g., *0*0*001, darts in border top, right
  MASK_V, C,   R,   0b010,    -1,  0,    #  e.g., *1*0*010, darts in border bottom, left
  MASK_V, C,   R,   0b011,    +1,  0,    #  e.g., *1*0*011, darts in border bottom, right

  MASK_H, 0,   R,   0b100,     0, -1,    #  e.g., 0*0*0100, darts in border left, upper
  MASK_H, 0,   R,   0b101,     0,  1,    #  e.g., 0*0*0101, darts in border left, lower
  MASK_H, C,   R,   0b110,     0, -1,    #  e.g., 1*0*0110, darts in border right, upper
  MASK_H, C,   R,   0b111,     0,  1,    #  e.g., 1*0*0111, darts in border right, lower

  MASK_I, 0,   0,   0b000,     0, -1,    #  e.g., *****000, darts in interior 
  MASK_I, 0,   0,   0b111,     0,  1,    #  e.g., *****111, darts in interior 
  MASK_I, 0,   0,   0b001,    +1, -1,    #  e.g., *****001, darts in interior 
  MASK_I, 0,   0,   0b101,    -1, +1,    #  e.g., *****101, darts in interior 
  MASK_I, 0,   0,   0b010,     0,  0,    #  e.g., *****010, darts in interior 
  MASK_I, 0,   0,   0b110,     0,  0,    #  e.g., *****110, darts in interior 
  MASK_I, 0,   0,   0b011,     1,  0,    #  e.g., *****011, darts in interior 
  MASK_I, 0,   0,   0b100,    -1,  0,    #  e.g., *****100, darts in interior 
], dtype=int).reshape(-1,6)

# %%
assert nBits <= 32, "Current routines implement the z-code up to 32 bits only!" 

XAs = np.zeros ((24,2), dtype=TYPE_UINT)
for i,(mask,c,r,bbb,dx,dy) in enumerate (bitpatterns):
    XAs[i,0] = (interleave2(c,r) & mask) << 3 |   bbb
    XAs[i,1] =                     mask  << 3 | 0b111

DXs = np.array (bitpatterns[:,4]).astype (np.int8)
DYs = np.array (bitpatterns[:,5]).astype (np.int8)

NEWBITS = np.array ([bitpatterns [i^1,3] for i in range (24)]).astype (np.uint8)

# check: DYs and DXs of subsequent entries need to cancel out
assert np.all (DYs.reshape (-1,2).sum (axis=1) == 0)
assert np.all (DXs.reshape (-1,2).sum (axis=1) == 0)

# %%
# for dart d, finds the first matching index in patterns

# the argmin version is simpler but assumes
# 1. the minimum is 0 (and not anything positive)
# 2. argmin returns the arg of the 1st smallest element

#atch = lambda d: np.flatnonzero (np.bitwise_and (np.bitwise_xor (d,XAs[:,0]), XAs[:,1]) == 0)[0]
match = lambda d: np.argmin      (np.bitwise_and (np.bitwise_xor (d,XAs[:,0]), XAs[:,1]))

# %%
alpha_0 = lambda d: d^1  # bitwise XOR with 1, i.e., flip bit 0 
alpha_2 = lambda d: d^2  # bitwise XOR with 2, i.e., flip bit 1 

def alpha_1 (d):
    i = match (d)
    dx,dy,newbits = DXs[i],DYs[i],NEWBITS[i]
    x,y = deinterleave2(d >> 3)
    x+=dx; y+=dy; 
    return (interleave2(x,y) << 3) | newbits

# %% [markdown]
# ### set

# %%
# set of darts

D = set()
for y in range (R):
    for x in range (C):
        for i in range (8):
            D |= {interleave2 (x,y) << 3 | i}

y = R
for x in range (C):
    for i in range (4):
        D |= {interleave2 (x,y) << 3 | i}
    
x = C
for y in range (R):
    for i in range (4):
        D |= {interleave2 (x,y) << 3 | 0b100 | i}

#print(f'D -> {D}')
# %% [markdown]
# # towards an implicit Set

# %%
# check for membership

max_x = interleave2 (C,0) << 3
max_y = interleave2 (0,R) << 3

mask_x = 0xffffffff ^ 2**nBits-1 | (MASK_H << 3)
mask_y = 0xffffffff ^ 2**nBits-1 | (MASK_V << 3)

"""
print (f'''\
{mask_x:032b} ... x-mask for (at most) 32-bit integer numbers
{mask_y:032b} ... y-mask for (at most) 32-bit integer numbers
''')
"""


def is_in (d):
    return (
           d & mask_x <  max_x and d & mask_y <  max_y                    # x <  C and y <  R
        or d & mask_x <  max_x and d & mask_y == max_y and not d & 0b100  # x <  C and y == R and last 3 bits are in 0...3
        or d & mask_x == max_x and d & mask_y <  max_y and     d & 0b100  # x == C and y <  R and last 3 bits are in 4...7
    )

# %%
assert all (    is_in(d) for d in D)

for d in D | {dd for dd in range (len(D), len(D)+10050)}:
        if (d in D and not is_in(d)) or (not d in D  and is_in(d)):
            z = d >> 3
            x,y = deinterleave2 (z)
            #print (f'{d:4} {d:08b} {x} {y} {is_in (d)}')

# %% [markdown]
# ### involution checks

# %%
# test if alpha_1 is involution for all darts in D

assert all ([alpha_0(alpha_0(d)) == d for d in D])
assert all ([alpha_1(alpha_1(d)) == d for d in D])
assert all ([alpha_2(alpha_2(d)) == d for d in D])
assert all ([alpha_2(alpha_0(alpha_2(alpha_0(d)))) == d for d in D])


# use this loop to debug if the above fails
# for d in D:
#     if alpha_1(alpha_1(d)) != d:
#         print (d, alpha_1(d), alpha_1(alpha_1(d)))
#         print (deinterleave2(d>>3))

# %% [markdown]
# ### sanity check with array-based gmaps

# %%
from combinatorial.notebooks.combinatorial.gmaps import nGmap

# %%
A = np.full ((3,1+max(D)),-1)  # initialize with invalid darts (-1)

for d in range (A.shape[1]):
    if d in D:
        A[0,d] = d^1
        A[2,d] = d^2
        A[1,d] = alpha_1(d)

# %%
g = nGmap.from_alpha_array (A)

#g.print_alpha_table()

assert g.is_valid
assert g.no_0_cells == (R+1)*(C+1)
assert g.no_1_cells == (R+1)*C + (C+1)*R
assert g.no_2_cells ==  R*C + 1 # also background counted
assert g.no_ccs     == 1

# %% [markdown]
# # Plot

# %%
colormap = plt.cm.gist_rainbow
colormap = plt.cm.brg

# %%
plt.figure(figsize=(24,16),frameon=False)
# plt.tight_layout()

for d in D:
#     e,i = d // 4, d % 4
#     y,x,a = e2yxa (e,R,C)
    x,y = deinterleave2(d >> 3)
    a = d >> 2 & 1
    i = d & 0b011
    
    b = f'{d:08b}' # bin string
    text = f'{d} = {b[:-3]}-{b[-3]}-{b[-2:]}'
    #color = colormap((256//((R+1)*(C+1)))*(d//8)) # if y < R and x < C else '0.6'
    color = 'black'
    
    if a == 0:
        xoff,yoff = (i % 2) *0.5, (i // 2)*0.02 -0.01
        plt.plot ([x+0.02 + xoff, x+0.48+xoff],[y+yoff,y+yoff], color=color)
        plt.text(x+.25+xoff,y+6*yoff,text,verticalalignment='center',color=color,horizontalalignment='center',fontsize=8)
    if a == 1:
        yoff,xoff = (i % 2) *0.5, (i // 2)*0.02 -0.01
        plt.plot ([x+xoff,x+xoff],[y+0.02+yoff,y+0.48+yoff], color=color)
        plt.text(x+6*xoff,y+.25+yoff,text,verticalalignment='center',color=color,horizontalalignment='center',fontsize=8, rotation=90)
        
    if d % 8 == 0: # and d < 8*R*C:
        plt.text (x+0.5, y+0.45, fr'{b[:-3]} $\rightarrow ({b[-7]}{b[-5]}_2, {b[-8]}{b[-6]}{b[-4]}_2) \rightarrow ({y}, {x})$',
        verticalalignment='center',color=color,horizontalalignment='center',fontsize=8)

# 
X,Y = 0.5 + np.array ([deinterleave2 (d) for d in sorted ([interleave2(x,y) for x in range (C+1) for y in range (R+1)])][:-1]).T
plt.plot (X,Y,':',color='0.8')
plt.scatter (X,Y,s=100,color ='0.8') #= [colormap((256//((R+1)*(C+1)))*(d//8)) for d in sorted (D)[::8]])

plt.gca().set_aspect(1)
plt.xticks([])
plt.yticks([]);
plt.ylim (R+1.2,-.2)
plt.xlim (-.2,C+1.2)
plt.title (f'Encoding ${R}\\times{C}$ baselevel\nusing Morton codes and bit flips')
pass
plt.gca().axis('off')


plt.savefig(f'Morton_full_{R}x{C}.pdf')

# %% [markdown]
# # fast $\alpha_1$ with 2-level look-up tables
# 
# any dart will fall into one of the following 3 classes according how many $90^\circ$ rotations $\alpha_1$ causes in the Level-0 (i.e, grid).
# 
# - $3 \times 90^\circ$ rotations, outer corner dart
# - $2 \times 90^\circ$ rotations, outer edge dart
# - $1 \times 90^\circ$ rotations, interior dart
# 
# Each of these cases requires special care:
# 
# 1. what is the matching pattern to identify the case
# 1. how $\alpha_1$ increments the coordinates of the new anchor pixel
# 1. how $\alpha_1$ transforms the 3-rightmost bits

# %%
# reorder the original bitpatterns by last 3 bits
# i.e, we have 0,0,0,1,1,1,2,2,2.... for corner,shell,interior each

reorder = np.argsort (bitpatterns[:,3],kind='stable')

# LUT_M ... 8x3x2 match (masks) LUT for each of 8 possible darts
# corner    | shell     | interior  |
# XOR AND   | XOR AND   | XOR AND   |

LUT_M = XAs  [reorder].reshape (8,3,2)

# LUT_T ... 8x3x3 transformation LUT for each of 8 possible darts
# corner    | shell     | interior  |
# bbb dx dy | bbb dx dy | bbb dx dy |

LUT_T = bitpatterns[:,3:].copy()
LUT_T [:,0] = NEWBITS

LUT_T = LUT_T[reorder].reshape (8,3,3)

def alpha_1_fast (d):
    """
    The update of the dart `d` depends on 
    - its position within the North-East border ( 8 possibilities)
    - its location in the image:  
        0. outer corner
        1. outer shell
        2. interior
    """

    i8 = d & 0b111 
    # i3 retrieves the index 0,1,2 for cases corner, shell, interior 
    i3 = np.count_nonzero((d ^ LUT_M[i8,:,0]) & LUT_M[i8,:,1])

    newbits,dx,dy = LUT_T [i8,i3]
    x,y = deinterleave2(d >> 3)
    x+=dx; y+=dy; 
    return (interleave2(x,y) << 3) | newbits
    
# check with the slower version
assert all (alpha_1_fast(d) == alpha_1(d) for d in D)

# involution check
assert all ([alpha_1_fast(alpha_1_fast(d)) == d for d in D])

#%timeit alpha_1(15)

#%timeit alpha_1_fast(15)

# %% [markdown]
# For a fixed $R \times C$ image we can pre-compute the matching patterns.

# %%
LUT_M;

# %% [markdown]
# The transformation patterns are independent of $R$, $C$.
# They depend only on how $\alpha_1$ in the upper-left corner was defined.
# In our case if was $\alpha_1(0) = 4$ and vice versa.

# %%
np.roll(LUT_T,2,axis=-1).reshape (8,9);

# %% [markdown]
# The three $(\Delta x,\Delta y,b^*)$  groups correspond to the 3 scenarios (270-outer-corner, 180-outer-edge, 90-interior)
# 
# |  b  ||  $\Delta x$ |  $\Delta y$ | $b^*$ || $\Delta x$  | $\Delta y$ | $b^*$  || $\Delta x$ | $\Delta y$ | $b^*$  |
# | -- || -- | -- | --|| -- | -- | --|| --| -- | --|
# | 0  ||  0 |  0 | 4 || -1 |  0 | 1 || 0 | -1|  7 |
# | 1  ||  1 |  0 | 6 ||  1 |  0 | 0 || 1 | -1|  5 |
# | 2  ||  0 | -1 | 5 || -1 |  0 | 3 || 0 |  0|  6 |
# | 3  ||  1 | -1 | 7 ||  1 |  0 | 2 || 1 |  0|  4 |
# | 4  ||  0 |  0 | 0 ||  0 | -1 | 5 ||-1 |  0|  3 |
# | 5  ||  0 |  1 | 2 ||  0 |  1 | 4 ||-1 |  1|  1 |
# | 6  || -1 |  0 | 1 ||  0 | -1 | 7 || 0 |  0|  2 |
# | 7  || -1 |  1 | 3 ||  0 |  1 | 6 || 0 |  1|  0 |

# %%
# %%

def plot(D):
    plt.figure(figsize=(24,16),frameon=False)
    # plt.tight_layout()

    for d in D:
        print(d)
    #     e,i = d // 4, d % 4
    #     y,x,a = e2yxa (e,R,C)
        x,y = deinterleave2(d >> 3)
        a = d >> 2 & 1
        i = d & 0b011
        
        b = f'{d:08b}' # bin string
        text = f'{d} = {b[:-3]}-{b[-3]}-{b[-2:]}'
        #color = colormap((256//((R+1)*(C+1)))*(d//8)) # if y < R and x < C else '0.6'
        color = 'black'
        
        if a == 0:
            xoff,yoff = (i % 2) *0.5, (i // 2)*0.02 -0.01
            plt.plot ([x+0.02 + xoff, x+0.48+xoff],[y+yoff,y+yoff], color=color)
            plt.text(x+.25+xoff,y+6*yoff,text,verticalalignment='center',color=color,horizontalalignment='center',fontsize=10)
        if a == 1:
            yoff,xoff = (i % 2) *0.5, (i // 2)*0.02 -0.01
            plt.plot ([x+xoff,x+xoff],[y+0.02+yoff,y+0.48+yoff], color='white')
            plt.text(x+6*xoff,y+.25+yoff,text,verticalalignment='center',color=color,horizontalalignment='center',fontsize=10, rotation=90)
            
        if d % 8 == 0: # and d < 8*R*C:
            plt.text (x+0.5, y+0.45, fr'{b[:-3]} $\rightarrow ({b[-7]}{b[-5]}_2, {b[-8]}{b[-6]}{b[-4]}_2) \rightarrow ({y}, {x})$',
            verticalalignment='center',color=color,horizontalalignment='center',fontsize=10)

    # 
    X,Y = 0.5 + np.array ([deinterleave2 (d) for d in sorted ([interleave2(x,y) for x in range (C+1) for y in range (R+1)])][:-1]).T
    plt.plot (X,Y,':',color='0.8')
    plt.scatter (X,Y,s=100,color ='0.8') #= [colormap((256//((R+1)*(C+1)))*(d//8)) for d in sorted (D)[::8]])

    plt.gca().set_aspect(1)
    plt.xticks([])
    plt.yticks([]);
    plt.ylim (R+1.2,-.2)
    plt.xlim (-.2,C+1.2)
    plt.title (f'Encoding ${R}\\times{C}$ baselevel\nusing Morton codes and bit flips')
    pass
    plt.gca().axis('off')


    #plt.savefig(f'Morton_full_{R}x{C}.pdf')
    plt.savefig(f'Morton_full_{R}x{C}.png')



def plot_chessboard(D, chessboard=None):
    plt.figure(figsize=(24,16),frameon=False)
    # plt.tight_layout()
    
    for d in D:
    #     e,i = d // 4, d % 4
    #     y,x,a = e2yxa (e,R,C)
        x,y = deinterleave2(d >> 3)
        a = d >> 2 & 1
        i = d & 0b011
        
        b = f'{d:08b}' # bin string
        text = f'{d} = {b[:-3]}-{b[-3]}-{b[-2:]}'
        #color = colormap((256//((R+1)*(C+1)))*(d//8)) # if y < R and x < C else '0.6'
        
        if chessboard != None: color = chessboard.labels[d]
        else: color = 'black'
        
        if a == 0:
            xoff,yoff = (i % 2) *0.5, (i // 2)*0.02 -0.01
            plt.plot ([x+0.02 + xoff, x+0.48+xoff],[y+yoff,y+yoff], color=color)
            plt.text(x+.25+xoff,y+6*yoff,text,verticalalignment='center',color='black',horizontalalignment='center',fontsize=8)
        if a == 1:
            yoff,xoff = (i % 2) *0.5, (i // 2)*0.02 -0.01
            plt.plot ([x+xoff,x+xoff],[y+0.02+yoff,y+0.48+yoff], color=color)
            plt.text(x+6*xoff,y+.25+yoff,text,verticalalignment='center',color='black',horizontalalignment='center',fontsize=8, rotation=90)
            
        if d % 8 == 0: # and d < 8*R*C:
            plt.text (x+0.5, y+0.45, fr'{b[:-3]} $\rightarrow ({b[-7]}{b[-5]}_2, {b[-8]}{b[-6]}{b[-4]}_2) \rightarrow ({y}, {x})$',
            verticalalignment='center',color='green',horizontalalignment='center',fontsize=8)

    # 
    X,Y = 0.5 + np.array ([deinterleave2 (d) for d in sorted ([interleave2(x,y) for x in range (C+1) for y in range (R+1)])][:-1]).T
    plt.plot (X,Y,':',color='0.8')
    plt.scatter (X,Y,s=100,color ='0.8') #= [colormap((256//((R+1)*(C+1)))*(d//8)) for d in sorted (D)[::8]])

    plt.gca().set_aspect(1)
    plt.xticks([])
    plt.yticks([]);
    plt.ylim (R+1.2,-.2)
    plt.xlim (-.2,C+1.2)
    plt.title (f'Encoding ${R}\\times{C}$ baselevel\nusing Morton codes and bit flips')
    pass
    plt.gca().axis('off')


    #plt.savefig(f'Morton_full_{R}x{C}.pdf')
    plt.savefig(f'Morton_Chessboard_{R}x{C}.png')
    plt.savefig(f'Morton_Chessboard_{R}x{C}.pdf')