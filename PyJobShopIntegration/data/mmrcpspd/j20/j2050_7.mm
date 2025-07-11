************************************************************************
file with basedata            : md370_.bas
initial value random generator: 1531318862
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  161
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       16        2       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6  16
   3        3          2           9  10
   4        3          3           5  12  15
   5        3          3          17  18  20
   6        3          3           7   8  10
   7        3          2          11  18
   8        3          3          14  17  21
   9        3          1          13
  10        3          2          13  14
  11        3          2          12  14
  12        3          3          17  20  21
  13        3          2          15  20
  14        3          1          19
  15        3          1          18
  16        3          1          19
  17        3          1          19
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
  2      1     2       7    0    9   10
         2     7       0    7    8    6
         3     8       0    7    8    3
  3      1     2       0    4    6   10
         2     8       8    0    5    9
         3    10       7    0    5    8
  4      1     1       3    0    8    5
         2     3       2    0    6    2
         3     8       2    0    5    1
  5      1     1       0    6    6    3
         2     4       0    5    2    2
         3     4       8    0    5    2
  6      1     1       7    0    6    9
         2     3       4    0    6    7
         3     9       4    0    6    6
  7      1     2       0    6    6    3
         2     2       4    0    5    3
         3     8       0    4    5    2
  8      1     2       8    0    5    8
         2     6       6    0    3    5
         3     8       5    0    2    1
  9      1     6       3    0    8    2
         2     7       0    7    8    2
         3     9       0    7    8    1
 10      1     2       0    5    9    3
         2     2       0    7    7    4
         3     6       0    3    5    2
 11      1     5       6    0    6    7
         2     6       0    4    5    6
         3    10       0    3    5    6
 12      1     1       0    7    8    8
         2     4       7    0    4    8
         3    10       3    0    1    7
 13      1     2       5    0    9    7
         2     8       0    4    8    7
         3     9       2    0    7    7
 14      1     2       8    0    9    5
         2     3       7    0    9    3
         3     7       0    9    7    3
 15      1     1       0    4    5    5
         2     2       0    4    5    4
         3     9       0    4    5    1
 16      1     2       4    0    4    7
         2     2       0    5    4    6
         3     5       0    1    3    4
 17      1     4       0    5    5    3
         2     7       7    0    5    2
         3    10       0    4    4    1
 18      1     1       0    7    1    7
         2     4       0    5    1    6
         3    10       6    0    1    5
 19      1     1       0   10    8    7
         2     2       5    0    6    6
         3    10       0    8    5    4
 20      1     1       0    4    9    7
         2     2       7    0    7    6
         3     3       7    0    7    4
 21      1     3       0    8    6    6
         2     4       0    7    5    6
         3     8       0    7    3    2
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   19   16  123  110
************************************************************************
DEADLINES:
jobnr.  deadline
  3       25
  5       1
  6       72
  7       10
  9       8
  11      43
  12      116
  13      38
  14      127
  15      39
  17      13
  18      47
  19      104
  21      161
************************************************************************
