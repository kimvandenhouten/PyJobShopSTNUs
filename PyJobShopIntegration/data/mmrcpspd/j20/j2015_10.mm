************************************************************************
file with basedata            : md335_.bas
initial value random generator: 1869607587
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  148
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       24        3       24
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6  13  14
   3        3          1           5
   4        3          3           8  12  14
   5        3          3           9  13  15
   6        3          3           7  17  18
   7        3          3           9  12  20
   8        3          3           9  11  15
   9        3          1          10
  10        3          1          21
  11        3          2          13  20
  12        3          1          21
  13        3          2          17  21
  14        3          3          16  17  18
  15        3          1          18
  16        3          2          19  20
  17        3          1          19
  18        3          1          19
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       8    7    5    0
         2     3       7    6    3    0
         3     8       7    6    0    5
  3      1     1       9    8    6    0
         2     2       6    7    0    6
         3     7       6    6    5    0
  4      1     3       8    6    0    8
         2     4       7    6    9    0
         3     5       7    5    0    7
  5      1     4       6    9    8    0
         2     4       7    9    0    4
         3     6       5    7    9    0
  6      1     4       8    8    0    8
         2     4       8    8    6    0
         3     8       7    7    0    8
  7      1     2       7   10    3    0
         2     4       7    8    0    7
         3     8       6    8    0    4
  8      1     2       5    7    8    0
         2     8       4    5    0    8
         3    10       3    4    0    8
  9      1     4       6    7    0    2
         2     5       6    3    0    2
         3     5       5    5    0    2
 10      1     7       8    5    5    0
         2     8       6    5    0    3
         3     9       6    4    0    3
 11      1     3       7    9    8    0
         2     5       6    9    6    0
         3     8       5    9    5    0
 12      1     1       9    7    0    6
         2     9       7    6    6    0
         3    10       2    3    0    3
 13      1     3       4    2    0    8
         2     4       2    2    0    6
         3     5       2    2    4    0
 14      1     2      10    6    5    0
         2     5       9    6    0    7
         3     9       9    6    0    5
 15      1     1       3    3    4    0
         2     3       2    3    0    5
         3     9       2    3    4    0
 16      1     1       7    6    0    9
         2     3       6    6    7    0
         3     3       6    6    0    7
 17      1     5       7    8    7    0
         2     6       6    8    5    0
         3     7       4    6    0    8
 18      1     1       3    2    9    0
         2     5       3    2    7    0
         3     7       2    2    6    0
 19      1     1       8    3    6    0
         2     9       8    3    5    0
         3    10       4    2    2    0
 20      1     2       4    8    0    6
         2     3       3    5    0    5
         3     4       3    3    0    3
 21      1     5       6    8    2    0
         2     5       7    7    0    7
         3    10       5    6    2    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   25   25   61   56
************************************************************************
DEADLINES:
jobnr.  deadline
  2       21
  7       147
  8       7
************************************************************************
