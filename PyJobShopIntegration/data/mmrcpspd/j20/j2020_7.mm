************************************************************************
file with basedata            : md340_.bas
initial value random generator: 893806981
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  162
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       25       17       25
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           8   9  15
   3        3          3           5   7  10
   4        3          3           6   7  14
   5        3          1          12
   6        3          2          12  17
   7        3          3          13  16  17
   8        3          3          11  13  16
   9        3          1          14
  10        3          3          11  15  16
  11        3          2          14  18
  12        3          1          13
  13        3          2          20  21
  14        3          2          19  20
  15        3          1          17
  16        3          1          20
  17        3          1          18
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
  2      1     1       0    4    0    6
         2     5      10    0    7    0
         3     8       8    0    0    3
  3      1     4      10    0    5    0
         2     5       8    0    3    0
         3     8       0    2    3    0
  4      1     6       5    0    2    0
         2     7       2    0    1    0
         3     8       0   10    0    3
  5      1     5       0    6    0    5
         2     6       7    0   10    0
         3     7       3    0    7    0
  6      1     4       7    0    7    0
         2     9       4    0    1    0
         3     9       0    8    1    0
  7      1     3       4    0    0    8
         2     4       0    3    0    4
         3     6       0    1    5    0
  8      1     8       0    6    0    8
         2     8       0    8    8    0
         3    10       0    5    0    8
  9      1     5       0    5    0    4
         2     8       9    0    8    0
         3     9       0    1    0    3
 10      1     1       0    5    0    5
         2     4       4    0    0    4
         3     6       0    1    6    0
 11      1     4       5    0    3    0
         2     5       0    6    0   10
         3     6       3    0    0    9
 12      1     1       0    9    7    0
         2     7       0    8    0    2
         3    10       4    0    6    0
 13      1     7       0    1    3    0
         2     8       3    0    2    0
         3    10       3    0    1    0
 14      1     2       0    7    0    5
         2     4       7    0    4    0
         3     5       7    0    0    3
 15      1     4       0    4    6    0
         2     4       0    4    0    9
         3     8       7    0    0    6
 16      1     5       0    2    7    0
         2     5       0    2    0    5
         3    10       0    1    7    0
 17      1     3       7    0    0    8
         2     9       7    0    0    7
         3    10       6    0    0    7
 18      1     1       5    0    5    0
         2     5       4    0    2    0
         3     6       4    0    0    5
 19      1     3       0    6    5    0
         2     7       0    5    0    9
         3     8       0    3    0    9
 20      1     7      10    0    7    0
         2     9       0    9    0    9
         3    10      10    0    5    0
 21      1     4       9    0    5    0
         2     6       0   10    5    0
         3     8       0    7    4    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   34   30   85   78
************************************************************************
DEADLINES:
jobnr.  deadline
  2       11
  3       156
  4       162
  5       136
  6       39
  7       7
  8       75
  9       32
  10      30
  11      57
  12      118
  13      7
  14      29
  15      10
  17      32
  18      65
  19      23
  20      72
  21      57
************************************************************************
