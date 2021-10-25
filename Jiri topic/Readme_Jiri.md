# Data structures for implicitly encoded n-Gmap-based pyramids

supervisors: Jiri Hladuvka, Walter Kropatsch

Cartesian grids (images, volumes) at the base level of a combinatorial pyramid can be encoded implicitly, i.e., both the dart-set and involutions require *O(1)* memory. While many involutions still remain implicit as cell are removed or contracted in the higher levels of the pyramid, increasing amounts of explicitly encoded involutions will be stored.

The straightforward way to implement involutions is an array storing for each dart its involution-counterparts. This, however, turns to be (1) prohibitive for huge maps resulting from large volumes and (2) not necessary given that large portion of the involutions can be encoded implicitly.

The goal is to propose and implement a data structure for storage and processing huge (50 to 400 Gigadarts) combinatorial pyramids on supercomputers provided by [Vienna Scientific Cluster](https://vsc.ac.at/).

## Deliverables

The following is a rough specifications what needs to be handed-in, before leaving PRIP, in order to get ECTS/signatures/certificates.

By the end of September:

* Detailed specification on what is to be done, emerging after discussions with the supervisor.

By the end of the exchange, i.e, before leaving Vienna.

* Technical report
* Documented software, ideally pushed to our gitlab server

## Resources

All the materials are available [here](https://portal.prip.tuwien.ac.at/nextcloud/index.php/s/8nroMBEesLYAmfb).


Get familiar with combinatorial maps before coming to Vienna. 
* Guillaume Damiand, Pascal Lienhardt. Combinatorial maps: Efficient data structures for computer graphics and image processing.
  * Pay attention to the following chapters of the book:\
    3\. Intuitive Presentation: Open & Closed n-maps, nG-maps. differences\
    4\. n-Gmaps: Dive into nGmaps\
    6\. Removal and Contraction
* Jiri Hladuvka. Encoding 2-gMaps using Morton codes and bit flips. Memo. April. 2021. See `JH_Morton_codes.pdf`

* Florian Bogner. Implicit Encoding of Contracted Cell nGmaps. Memo. August. 2021. See `FB_implicit encoding of contracted nGmaps.pdf` in materials.

### Some software libraries

- combinatorial.zip: Python libraries exported from PRIP GitLab
- [bidict library](https://bidict.readthedocs.io/) in Python
- [Boost bimap library](https://www.boost.org/doc/libs/1_75_0/libs/bimap/doc/html/index.html) in C++