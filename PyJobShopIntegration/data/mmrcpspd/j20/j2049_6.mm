************************************************************************
file with basedata            : md369_.bas
initial value random generator: 253013575
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  158
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       26       17       26
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   9  10
   3        3          3           6  10  12
   4        3          1          10
   5        3          3           8  14  18
   6        3          3           7  16  17
   7        3          2           9  15
   8        3          1          11
   9        3          2          14  18
  10        3          2          16  17
  11        3          1          12
  12        3          3          13  16  17
  13        3          2          15  20
  14        3          2          19  20
  15        3          2          19  21
  16        3          1          21
  17        3          2          19  20
  18        3          1          21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       0   10    4    6
         2     2       5    0    4    8
         3    10       5    0    4    5
  3      1     3       0    9    5    5
         2     3       3    0    6    6
         3    10       0    8    3    2
  4      1     1       0    4    7    8
         2     6       7    0    3    8
         3     7       0    4    3    7
  5      1     2       6    0    8    5
         2     5       6    0    6    5
         3    10       0    9    6    2
  6      1     1       8    0    8   10
         2     8       5    0    8    6
         3     9       4    0    5    2
  7      1     1       0    5    7    6
         2     5       4    0    6    6
         3     5       0    5    6    6
  8      1     2       0    9   10    8
         2     3       0    5    5    3
         3     9       5    0    3    1
  9      1     3       7    0   10   10
         2     8       4    0    7    8
         3    10       2    0    7    6
 10      1     2       6    0    4    5
         2     6       0    3    3    4
         3     7       4    0    3    3
 11      1     1       0    9    5    8
         2     1       0    9    9    7
         3     8       3    0    3    6
 12      1     5       0    6    6    8
         2     7       0    5    5    8
         3    10       9    0    4    8
 13      1     5       9    0    6    5
         2     6       0    5    6    5
         3     8       0    5    6    4
 14      1     1       7    0    7    8
         2     2       0    7    6    6
         3     4       0    6    5    5
 15      1     1       0    9    4    4
         2     1       7    0    5    4
         3     3       5    0    3    3
 16      1     1       2    0    4   10
         2     1       0    6    4    9
         3    10       0    5    2    8
 17      1     2       3    0    3    2
         2     2       0    9    3    1
         3     2       0    8    4    1
 18      1     2       7    0    7    8
         2     9       0    5    7    5
         3     9       4    0    7    7
 19      1     2       0    8    9    6
         2     6       0    7    8    6
         3     8       0    4    7    1
 20      1     6       7    0   10    8
         2     9       6    0   10    6
         3     9       0    7   10    6
 21      1     8       6    0    3    6
         2     8       0    4    4    6
         3    10       0    3    1    6
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    8    9  124  126
************************************************************************
DEADLINES:
jobnr.  deadline
  2       68
  5       147
  6       144
  8       102
  10      47
  12      110
  17      104
  19      73
  20      114
************************************************************************
