************************************************************************
file with basedata            : md373_.bas
initial value random generator: 821065833
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  143
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       18        5       18
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           7   9  11
   3        3          3          16  19  21
   4        3          3           5   8  13
   5        3          2           6  14
   6        3          3          12  18  21
   7        3          3           8  12  17
   8        3          1          15
   9        3          3          10  14  15
  10        3          1          18
  11        3          2          12  15
  12        3          1          16
  13        3          1          16
  14        3          1          17
  15        3          2          19  20
  16        3          1          20
  17        3          2          18  21
  18        3          2          19  20
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       8    3    7    6
         2     3      10    4    5    8
         3     4       8    3    3    5
  3      1     1       4    4    6    6
         2     3       3    3    6    6
         3     8       1    3    5    6
  4      1     2       8    4   10    2
         2     5       5    2    8    2
         3     7       4    2    8    2
  5      1     2       5    4    7    3
         2     4       4    3    6    2
         3     5       4    3    4    2
  6      1     6       7    8   10   10
         2     8       6    4    7    9
         3     9       5    1    5    9
  7      1     5       6    7    3    4
         2     5       6    6    4    5
         3     6       5    6    2    3
  8      1     4       6    9    7    6
         2     5       5    8    7    5
         3     9       5    8    5    5
  9      1     3       5    4    7    7
         2     8       2    4    6    3
         3     8       3    4    5    4
 10      1     1       4    7   10    7
         2     3       4    7    8    6
         3     4       2    4    3    4
 11      1     1       9    4    8    5
         2     2       5    4    6    5
         3     3       2    4    6    4
 12      1     1       7    6    9    8
         2     4       6    4    7    7
         3     5       4    3    4    5
 13      1     5       7    4    8    7
         2     6       5    4    6    5
         3     9       3    3    6    4
 14      1     3       7    6    8    9
         2     4       4    5    8    8
         3     7       1    4    5    7
 15      1     1       8    7    7    4
         2     9       8    5    6    3
         3     9       8    5    5    4
 16      1     4       8    8    6    6
         2     6       5    8    6    5
         3     6       3    8    6    6
 17      1     4       7    2    4    8
         2     9       6    1    4    4
         3    10       2    1    4    2
 18      1     2       8    4    9    7
         2     3       4    4    7    7
         3     8       2    4    3    7
 19      1     2       6    6    9    9
         2     7       6    5    8    6
         3     9       6    4    5    3
 20      1     3       7    1    7    3
         2     6       4    1    6    2
         3     8       3    1    4    1
 21      1     4       7    9    5    6
         2     5       6    8    5    6
         3     9       6    5    5    5
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   14   14  134  116
************************************************************************
DEADLINES:
jobnr.  deadline
  2       3
  3       116
  4       133
  9       88
  11      2
  15      29
  16      58
  17      55
************************************************************************
