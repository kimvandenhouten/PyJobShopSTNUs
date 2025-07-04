************************************************************************
file with basedata            : md351_.bas
initial value random generator: 323618701
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  171
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       26       16       26
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3          13  14  16
   3        3          2           6   9
   4        3          2           5  14
   5        3          2           7  16
   6        3          2           8  16
   7        3          3           8  17  18
   8        3          1          10
   9        3          3          10  12  18
  10        3          1          11
  11        3          1          13
  12        3          2          13  14
  13        3          2          15  20
  14        3          2          15  17
  15        3          2          19  21
  16        3          3          18  19  20
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
  2      1     5       3    5    6    0
         2     9       3    5    3    0
         3     9       3    4    0   10
  3      1     5       8    2    5    0
         2     7       8    2    4    0
         3     9       6    1    4    0
  4      1     1       6    9   10    0
         2     5       5    9    0    7
         3     9       4    8    0    3
  5      1     1       6    7    0    5
         2     1       6    7    8    0
         3     8       6    6    6    0
  6      1     6       6    5    0    7
         2     8       6    4    0    7
         3    10       4    3    0    6
  7      1     2       7    6    4    0
         2     4       6    4    4    0
         3     5       1    2    0   10
  8      1     3       4    7    0    8
         2     6       4    6    0    5
         3     9       3    6    0    2
  9      1     6       6    6    0    3
         2     9       6    6    1    0
         3     9       6    5    0    2
 10      1     1       4    9    0    9
         2     3       3    8    6    0
         3     3       3    7    0    2
 11      1     1       6    9    0    9
         2     8       4    9    9    0
         3     9       4    6    9    0
 12      1     2       8    6    0    8
         2     2       9    8    4    0
         3     8       7    3    0    8
 13      1     3       4    7    6    0
         2     4       3    7    0    8
         3     6       3    6    0    4
 14      1     4       8    7    0    7
         2     6       8    6    0    7
         3    10       6    5    4    0
 15      1     1       5    2    0    3
         2     6       5    1    2    0
         3     9       5    1    0    2
 16      1     2       5    9    0    7
         2     7       4    4    7    0
         3    10       3    3    4    0
 17      1     5       8    9    0    4
         2     6       6    8    0    2
         3    10       4    8    3    0
 18      1     3       3    5    0    1
         2     6       2    4    4    0
         3    10       2    2    0    1
 19      1     4       4    7    0    5
         2     6       3    7    0    4
         3     8       3    7    0    3
 20      1     1       9    5    0    8
         2     4       8    5    1    0
         3    10       8    2    0    5
 21      1     2       6    5    0    2
         2     3       3    4   10    0
         3    10       2    2    9    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   20   90  121
************************************************************************
DEADLINES:
jobnr.  deadline
  2       112
  4       30
  5       33
  6       84
  7       51
  9       16
  10      25
  11      54
  13      160
  14      70
  15      11
  17      62
  18      78
  19      65
  20      131
  21      99
************************************************************************
