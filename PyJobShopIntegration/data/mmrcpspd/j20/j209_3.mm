************************************************************************
file with basedata            : md329_.bas
initial value random generator: 1070424380
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
    1     20      0       21       10       21
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           7   8  11
   3        3          3           5   9  18
   4        3          3          11  15  18
   5        3          3           6  10  21
   6        3          3           7  11  13
   7        3          1          15
   8        3          2          13  14
   9        3          2          20  21
  10        3          2          12  16
  11        3          1          12
  12        3          2          19  20
  13        3          3          15  16  19
  14        3          1          17
  15        3          1          17
  16        3          1          17
  17        3          1          20
  18        3          2          19  21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       9    0    2    0
         2     7       0    3    0    8
         3     7       0    5    1    0
  3      1     1       0    4    0    3
         2     5       0    3    7    0
         3    10       2    0    1    0
  4      1     1       3    0    5    0
         2     3       0    4    4    0
         3     4       2    0    0    8
  5      1     4       0    9    0    9
         2     7       4    0    0    8
         3     9       0    9    9    0
  6      1     1       5    0    0    8
         2     2       4    0    6    0
         3    10       0    9    5    0
  7      1     4      10    0    5    0
         2     7       0    9    4    0
         3     8       0    8    0   10
  8      1     1       4    0    0    7
         2     1       8    0    6    0
         3     7       0    3    0    7
  9      1     1       6    0    0    8
         2     5       0    7    3    0
         3     8       6    0    0    4
 10      1     2       6    0    9    0
         2    10       0    5    6    0
         3    10       5    0    0    4
 11      1     2       8    0    0    8
         2     5       5    0    5    0
         3     6       0    3    0    8
 12      1     1       5    0    0    3
         2     4       0    6    0    1
         3     6       5    0    3    0
 13      1     1       0    3    0    6
         2     4       2    0   10    0
         3     8       0    2    0    5
 14      1     1       0    1    0   10
         2     8       4    0    0    4
         3    10       1    0    3    0
 15      1     5       9    0    0    3
         2     8       9    0    0    2
         3     9       8    0    8    0
 16      1     1       0    9    0    6
         2     3       0    9    6    0
         3     8       2    0    0    4
 17      1     2       0    1    3    0
         2     7       0    1    0    5
         3     8       0    1    0    4
 18      1     3       6    0    0    8
         2     7       0    8    0    8
         3     8       5    0    0    8
 19      1     3       0    6    0    6
         2     3       0    5   10    0
         3     7       4    0    0    4
 20      1     4       7    0    0    7
         2     5       0    9    0    7
         3     5       5    0    0    7
 21      1     7       2    0    5    0
         2     8       2    0    3    0
         3    10       0    6    0    2
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   13    9   53   72
************************************************************************
DEADLINES:
jobnr.  deadline
  2       10
  3       26
  8       125
  9       26
  11      51
  13      12
  16      131
  18      10
  20      131
************************************************************************
