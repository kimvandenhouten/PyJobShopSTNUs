************************************************************************
file with basedata            : md345_.bas
initial value random generator: 881875105
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  154
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       23       16       23
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           6  10
   3        3          2          10  12
   4        3          3           5  11  16
   5        3          1          10
   6        3          3           7  16  19
   7        3          3           8   9  11
   8        3          3          13  14  15
   9        3          3          12  17  18
  10        3          3          13  18  19
  11        3          2          12  15
  12        3          1          14
  13        3          2          20  21
  14        3          1          20
  15        3          1          17
  16        3          1          18
  17        3          2          20  21
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
  2      1     3       3    0    0    5
         2     5       2    0    5    0
         3     8       2    0    4    0
  3      1     7       2    0    0    5
         2     7       2    0    2    0
         3     9       2    0    0    4
  4      1     6       8    0    0    8
         2     7       0    8    4    0
         3     8       6    0    0    1
  5      1     3       8    0    8    0
         2     6       0    5    6    0
         3     9       0    5    0    1
  6      1     1       0    5    8    0
         2     5       0    5    0    4
         3     5       3    0    0    5
  7      1     2       9    0    5    0
         2     6       7    0    0    7
         3     6       5    0    4    0
  8      1     1       0    4    7    0
         2     3       0    2    4    0
         3     4       3    0    0    6
  9      1     5       0    4    7    0
         2     7       8    0    3    0
         3     8       0    4    0    6
 10      1     2       3    0    8    0
         2     7       0    4    0    7
         3     9       2    0    7    0
 11      1     1       0    1    5    0
         2     6       9    0    0    4
         3     7       7    0    0    2
 12      1     2       2    0    2    0
         2     6       0    7    2    0
         3    10       0    5    1    0
 13      1     5       0    8    0    8
         2     7       9    0    0    8
         3     7       0    1    0    8
 14      1     3       0    8    7    0
         2     9       6    0    0    4
         3    10       5    0    5    0
 15      1     8       7    0    9    0
         2     8       0    5    6    0
         3     8       0    5    0    6
 16      1     2       4    0    5    0
         2     5       2    0    3    0
         3    10       0    7    0    6
 17      1     2       7    0    0    6
         2     5       0    4    5    0
         3     5       4    0    0    6
 18      1     2       0    7    0   10
         2     6       9    0    9    0
         3     8       7    0    8    0
 19      1     4       0   10    2    0
         2     6       0   10    0   10
         3     7       9    0    0    8
 20      1     6       0    9   10    0
         2     6       0    7    0    7
         3     8       6    0    9    0
 21      1     1       7    0    7    0
         2     5       0    1    0   10
         3     8       3    0    0   10
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   13    7  115  121
************************************************************************
DEADLINES:
jobnr.  deadline
  2       53
  3       6
  4       100
  7       20
  8       149
  10      69
  13      86
  15      135
  16      19
  17      135
  19      65
  20      10
  21      112
************************************************************************
